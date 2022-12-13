import gym
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

"""
set init to xavier

"""


class agent():
    def __init__(self, s0, act_space_high, act_space_low, obs_dim, state_space_samples):
        """conveniece"""
        self.my_path = os.getcwd()
        self.render_size = 10  # render every 10 steps

        """parameter"""
        self.gamma = 0.97
        self.batch_size =64

        """state"""
        self.s0 = s0
        self.obs_dim = obs_dim

        """action space"""
        self.action_space_low = [(float(str(
            act_space_low)))]  # work around in order to process every environment's action space input
        self.action_space_high = [(float(str(act_space_high)))]
        #print(self.action_space_high,self.action_space_low, type(self.action_space_high),"\n")

        """NNs"""
        self.a=self.actor(state_space_samples)
        self.c=self.critic(state_space_samples)

    """__METHODS__"""

    """reward and advantages"""

    def discounted_rewards_advantages(self, rewards, V_batch, V_nplus1, dones=0):
        discounted_rewards = np.array(rewards + [V_nplus1])
        for t in reversed(range(len(rewards))):
            discounted_rewards[t] = rewards[t] + self.gamma * \
                discounted_rewards[t+1] * (1-dones[t])
        discounted_rewards = discounted_rewards[:-1]
        # advantages are bootstrapped discounted rewards - values, using Bellman's equation ==> check out README for mathematics
        advantages = discounted_rewards - np.stack(V_batch)
        return discounted_rewards, advantages

    def action_value(self, state):  # pass state through actor and choose action
        norm_dist = self.a.predict_on_batch(state)  # runs agent.call()
        value = self.c.predict_on_batch(state)  # runs critic.call()
        # sample from prob distribution and remove batch dimension
        action = tf.squeeze(norm_dist.sample(1), axis=0)
        # fit into action space
        action = tf.clip_by_value(
            action, self.action_space_low, self.action_space_high)
        return action, value

    class actor(keras.Model):

        """__CONSTRUCTOR__"""

        def __init__(self, state_space_samples):
            super().__init__()

            """conveniece"""
            self.model_path = 0

            """parameters"""
            self.actor_loss_weight = 1
            self.entropy_loss_weight = 0.04

            """ NN"""
            self.state_space_samples = state_space_samples
            self.nodes_per_dense_layer = 60
            self.actor_weight_dict = {}

            # state normalisation
            self.actor_weight_dict["state_norm"] = layers.Normalization(axis=-1)
            self.actor_weight_dict["state_norm"].adapt(self.state_space_samples)
            # common layers
            self.actor_weight_dict["dense1"] = layers.Dense(
                self.nodes_per_dense_layer, activation=tfa.activations.mish, kernel_initializer=keras.initializers.glorot_normal())
            self.actor_weight_dict["dense2"] = layers.Dense(
                self.nodes_per_dense_layer, activation=tfa.activations.mish, kernel_initializer=keras.initializers.glorot_normal())
            # outlets
            self.actor_weight_dict["mu"] = layers.Dense(1)  # outlet 1
            self.actor_weight_dict["sigma_inbetween1"] = layers.Dense(1)
            self.actor_weight_dict["sigma"] = layers.Dense(
                1, activation=tf.nn.softplus)  # outlet 2 | sigma has to be positive

        def call(self, inputs):
            x = inputs
            x = self.actor_weight_dict["state_norm"](x)
            x = self.actor_weight_dict["dense1"](x)
            x = self.actor_weight_dict["dense2"](x)
            mu = self.actor_weight_dict["mu"](x)
            sigma = self.actor_weight_dict["sigma_inbetween1"](x)
            sigma = self.actor_weight_dict["sigma"](sigma)
            norm_dist = tfp.distributions.Normal(mu, sigma)
            return norm_dist  

        # lets penalize high uncertainty (== wide streched norm dist)
        def entropy_loss(self, norm_dist):
            return - norm_dist.entropy()

        """
        The loss works like the following:probability that the chosen action actually is taken chosen * advantage of that decision
        """
        def actor_loss(self, combined, norm_dist):
            actions = combined[:, 0]  # first column holds a_t
            advantages = combined[:, 1]  # second column holds A_t
            loss = -norm_dist.log_prob(actions)*advantages

            return loss * self.actor_loss_weight + self.entropy_loss_weight * self.entropy_loss(norm_dist)

    class critic(keras.Model):

        """__CONSTRUCTOR__"""

        def __init__(self, state_space_samples):
            super().__init__()

            """conveniece"""
            self.model_path = 0

            """parameters"""


            """ NN"""
            self.state_space_samples = state_space_samples
            self.nodes_per_dense_layer = 300
            self.critic_weight_dict = {}

            # state normalisation
            self.critic_weight_dict["state_norm"] = layers.Normalization(
                axis=-1)
            self.critic_weight_dict["state_norm"].adapt(
                self.state_space_samples)
            # common layers
            self.critic_weight_dict["dense1"] = layers.Dense(
                self.nodes_per_dense_layer, activation=tfa.activations.mish, kernel_initializer=keras.initializers.glorot_normal())
            self.critic_weight_dict["dense2"] = layers.Dense(
                self.nodes_per_dense_layer, activation=tfa.activations.mish, kernel_initializer=keras.initializers.glorot_normal())
            # outlet
            self.critic_weight_dict["value"] = layers.Dense(1)

        def call(self, inputs):
            x = inputs
            x = self.critic_weight_dict["state_norm"](x)
            x = self.critic_weight_dict["dense1"](x)
            x = self.critic_weight_dict["dense2"](x)
            value = self.critic_weight_dict["value"](x)
            return value

        def critic_loss(self, discounted_rewards, predicted_values):
            loss = keras.losses.mean_squared_error(
                discounted_rewards, predicted_values)
            return loss
