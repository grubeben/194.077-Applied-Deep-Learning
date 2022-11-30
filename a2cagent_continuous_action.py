import gym
import pybullet as p
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers
from datetime import datetime

import os
import os.path
import glob
import shutil
import json

#continuous task https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/a2c/a2c.py

class A2CAgent(keras.Model):

    """__CONSTRUCTOR__"""

    def __init__(self,s0, act_space_high, act_space_low, obs_dim, use_existing_policy=False):
        super().__init__()
        """conveniece"""
        self.my_path = os.getcwd()
        self.model_path=0

        """parameters"""
        self.critic_loss_weight = 0.4
        self.actor_loss_weight = 1
        self.entropy_loss_weight = 0.04
        self.batch_size = 64
        self.gamma = 0.95
        self.obs_dim = obs_dim

        """state"""
        self.s0 = s0

        """action space"""
        self.action_space_low=act_space_high
        self.action_space_high=act_space_low
        

        """ NN"""
        self.my_weight_dict = {}
        self.my_weight_dict["dense1"] = layers.Dense(64, activation=tfa.activations.mish, kernel_initializer=keras.initializers.he_normal())
        self.my_weight_dict["dense2"] = layers.Dense(64, activation=tfa.activations.mish, kernel_initializer=keras.initializers.he_normal())
        self.my_weight_dict["value"] = layers.Dense(1) #outlet 1
        self.my_weight_dict["mu"] = layers.Dense(1) #outlet 2
        self.my_weight_dict["sigma_inbetween"] = layers.Dense(1)
        self.my_weight_dict["sigma"] = layers.Dense(1,activation=tf.nn.softplus) #outlet 3 | sigma has to be positive 


    """__METHODS__"""

    """reward and advantages"""
    def discounted_rewards_advantages(self,rewards,V_batch,V_nplus1,dones=0):
        discounted_rewards = np.array(rewards + [V_nplus1])
        for t in reversed(range(len(rewards))):
            discounted_rewards[t] = rewards[t] + self.gamma * discounted_rewards[t+1] * (1-dones[t])
        discounted_rewards = discounted_rewards[:-1]
        # advantages are bootstrapped discounted rewards - values, using Bellman's equation
        advantages = discounted_rewards - np.stack(V_batch)
        return discounted_rewards, advantages

    """losses"""
    def critic_loss(self,discounted_rewards, predicted_values):
        return keras.losses.mean_squared_error(discounted_rewards, predicted_values) * self.critic_loss_weight

    # def pseudo_entropy_loss(self,policy_logits): # let's penalize large sigma?
    #     return -(keras.losses.categorical_crossentropy(policy_logits,policy_logits, from_logits=True) * self.entropy_loss_weight)

    def actor_loss(self,combined,_):
        loss=0

        for i in range(len(combined[:, 0])):
            action = combined[i, 0]  # first column holds a_t
            advantage = combined[i, 1]  # second column holds A_t
            mu = combined[i, 2] 
            sigma = combined[i, 3] 
            norm_dists=tfp.distributions.Normal(mu,sigma)
            loss+= -tf.math.log(norm_dists.prob(action)*advantage)
        loss= tf.math.reduce_mean(loss)
        
        print("\n\n", loss, "\n\n")
        return loss * self.actor_loss_weight

    """NN methods"""

    def call(self, inputs):  # pass state forward through NN
        x = self.my_weight_dict["dense1"](inputs)
        x = self.my_weight_dict["dense2"](x)
        value= self.my_weight_dict["value"](x)
        mu=self.my_weight_dict["mu"](x)

        sigma = self.my_weight_dict["sigma_inbetween"](x)
        sigma = self.my_weight_dict["sigma"](sigma)

        return value, mu, sigma

    def action_value(self, state):  # pass state through NN and choose action
        value, mu, sigma = self.predict_on_batch(state)  # runs call() from above
        norm_dist=tfp.distributions.Normal(mu,sigma)
        action = tf.squeeze(norm_dist.sample(1), axis=0) #sample from prob distribution and remove batch dimension
        action = tf.clip_by_value(action, self.action_space_low, self.action_space_high) #fit into action space
        return action, value, mu,sigma
