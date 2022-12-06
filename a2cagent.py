import gym
import pybullet as p
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras import layers
from datetime import datetime

import os
import os.path
import glob
import shutil
import json


class A2CAgent(keras.Model):

    """__CONSTRUCTOR__"""

    def __init__(self,s0, act_dim, obs_dim, use_existing_policy=False):
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
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        """state"""
        self.s0 = s0

        """ NN"""
        self.my_weight_dict = {}
        self.my_weight_dict["dense1"] = layers.Dense(64, activation=tfa.activations.mish, kernel_initializer=keras.initializers.he_normal())
        self.my_weight_dict["dense2"] = layers.Dense(64, activation=tfa.activations.mish, kernel_initializer=keras.initializers.he_normal())
        self.my_weight_dict["value"] = layers.Dense(1)
        self.my_weight_dict["policy_logits"] = layers.Dense(self.act_dim)

        # self.dense1 = layers.Dense(
        #     64, activation=tfa.activations.mish, kernel_initializer=keras.initializers.he_normal())
        # self.dense2 = layers.Dense(
        #     64, activation=tfa.activations.mish, kernel_initializer=keras.initializers.he_normal())
        # # Instead of creating two entire NNs we just use one with two outlet channels 
        # # only the last layer weights decide whether we obtain P(s,a) or V(s)
        # self.value = layers.Dense(1)  # outlet 1
        # self.policy_logits = layers.Dense(self.act_dim)  # outlet 2

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

    def entropy_loss(self,policy_logits):
        return -(keras.losses.categorical_crossentropy(policy_logits,policy_logits, from_logits=True) * self.entropy_loss_weight)

    def actor_loss(self,combined, policy_logits):
        actions = combined[:, 0]  # first column holds a_t
        advantages = combined[:, 1]  # second column holds A_t
        """
        plicy_loss = 0
        for i in range(len(actions)):
            policy_loss += advantages[i]*keras.losses.categorical_crossentropy(policy_logits[tf.cast(actions[i], tf.int32)])
        """
        sparse_ce = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        actions = tf.cast(actions, tf.int32)
        policy_loss = sparse_ce(actions, policy_logits, sample_weight=advantages)

        return 0 #policy_loss * self.actor_loss_weight + self.entropy_loss(policy_logits) * self.entropy_loss_weight

    """NN methods"""
    # def call(self, inputs):  # pass state forward through NN
    #     x = self.dense1(inputs)
    #     x = self.dense2(x)
    #     return self.value(x), self.policy_logits(x)

    def call(self, inputs):  # pass state forward through NN
        x = self.my_weight_dict["dense1"](inputs)
        x = self.my_weight_dict["dense2"](x)
        return self.my_weight_dict["value"](x), self.my_weight_dict["policy_logits"](x)

    def action_value(self, state):  # pass state through NN and choose action
        value, logits = self.predict_on_batch(state)  # runs call() from above
        action = tf.random.categorical(logits, 1)[0] # choose random action based on action probabilities
        return action, value