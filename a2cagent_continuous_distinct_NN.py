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


class agent():
    def __init__(self, s0, act_space_high, act_space_low, obs_dim, state_space_samples, activation_function, initializer, state_normalization, batch_normalization):
        """conveniece"""
        self.my_path = os.getcwd()
        self.tb_path = None
        self.render_size = 10  # render every 10 steps

        """parameter"""
        self.gamma = 0.95
        self.batch_size = 64

        """state"""
        self.s0 = s0
        self.obs_dim = obs_dim

        """action space"""
        """ work around in order to process every environment's action space input"""
        self.action_space_low = [(float(str(
            act_space_low)))]
        self.action_space_high = [(float(str(act_space_high)))]

        """NNs"""
        self.a = self.actor(state_space_samples, activation_function,
                            initializer, state_normalization, batch_normalization)
        self.c = self.critic(state_space_samples, activation_function,
                             initializer, state_normalization, batch_normalization)

    """__METHODS__"""

    """reward and advantages"""

    def discounted_rewards_advantages(self, rewards, V_batch, V_nplus1, dones=0):
        """
        computes discounted rewards and advantages from rewards collected, NN-value-function estimates (V(0-t)) during batch 
        and NN-value-function estimate (V(t+1)) for the step ahead
        """
        discounted_rewards = np.array(rewards + [V_nplus1])
        for t in reversed(range(len(rewards))):
            discounted_rewards[t] = rewards[t] + self.gamma * \
                discounted_rewards[t+1] * (1-dones[t])
        discounted_rewards = discounted_rewards[:-1]
        # advantages are bootstrapped discounted rewards - values, using Bellman's equation ==> check out README for mathematics
        advantages = discounted_rewards - np.stack(V_batch)
        return discounted_rewards, advantages

    def action_value(self, state):
        """
        choose action from normal_distribution(a)
        """
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

        def __init__(self, state_space_samples, activation_function, initializer, state_normalization, batch_normalization):
            super().__init__()

            """conveniece"""
            self.model_path = 0

            """parameters"""
            self.actor_loss_weight = 1
            self.entropy_loss_weight = 0.04

            """ NN"""
            # network configuration
            self.metadata = {
                'activation.functions':     {'relu': keras.activations.relu, 'mish': tfa.activations.mish},
                'initialization.functions': {'normal': keras.initializers.he_normal(), 'xavier': keras.initializers.glorot_normal()}
            }
            self.activation_function = self.metadata['activation.functions'][activation_function]
            self.initializer = self.metadata['initialization.functions'][initializer]
            self.state_space_samples = state_space_samples
            self.state_normalization = state_normalization
            self.batch_normalization = batch_normalization
            self.nodes_per_dense_layer = 60
            self.actor_weight_dict = {}

            # normalisation
            self.actor_weight_dict["state_norm"] = layers.Normalization(
                axis=-1)
            self.actor_weight_dict["state_norm"].adapt(
                self.state_space_samples)
            self.actor_weight_dict["batch_norm"] = layers.BatchNormalization()
            # common layers
            self.actor_weight_dict["dense1"] = layers.Dense(
                self.nodes_per_dense_layer, activation=self.activation_function, kernel_initializer=self.initializer)
            self.actor_weight_dict["dense2"] = layers.Dense(
                self.nodes_per_dense_layer, activation=self.activation_function, kernel_initializer=self.initializer)
            # outlets
            self.actor_weight_dict["mu"] = layers.Dense(1)  # outlet 1
            self.actor_weight_dict["sigma_inbetween1"] = layers.Dense(1)
            self.actor_weight_dict["sigma"] = layers.Dense(
                1, activation=tf.nn.softplus)  # outlet 2 | sigma has to be positive

        def call(self, inputs):
            """
            passes state forward through NN; method-name 'call()' can not be changed, since tf.keras uses this overload to build NN
            output: normal_distribution(a)
            """
            x = inputs
            if (self.state_normalization == True):
                x = self.actor_weight_dict["state_norm"](x)
            x = self.actor_weight_dict["dense1"](x)
            if (self.batch_normalization == True):
                x = self.actor_weight_dict["batch_norm"](x)
            x = self.actor_weight_dict["dense2"](x)
            mu = self.actor_weight_dict["mu"](x)
            sigma = self.actor_weight_dict["sigma_inbetween1"](x)
            sigma = self.actor_weight_dict["sigma"](sigma)
            norm_dist = tfp.distributions.Normal(mu, sigma)
            return norm_dist

        def entropy_loss(self, norm_dist):
            """
            lets penalize high uncertainty (== wide streched norm dist)
            """
            return - norm_dist.entropy()

        def actor_loss(self, combined, norm_dist):
            """
            The loss works like the following:
            probability that the chosen action actually is taken chosen * advantage of that decision
            """
            actions = combined[:, 0]  # first column holds a_t
            advantages = combined[:, 1]  # second column holds A_t
            loss = norm_dist.prob(actions)*advantages
            loss=tf.math.reduce_mean(loss)

            return loss * self.actor_loss_weight #+ self.entropy_loss_weight * self.entropy_loss(norm_dist)

    class critic(keras.Model):

        """__CONSTRUCTOR__"""

        def __init__(self, state_space_samples, activation_function, initializer, state_normalization, batch_normalization):
            super().__init__()

            """convenience"""
            self.model_path = 0

            """ NN"""
            self.metadata = {
                'activation.functions':     {'relu': keras.activations.relu, 'mish': tfa.activations.mish},
                'initialization.functions': {'normal': keras.initializers.he_normal(), 'xavier': keras.initializers.glorot_normal()}
            }
            self.activation_function = self.metadata['activation.functions'][activation_function]
            self.initializer = self.metadata['initialization.functions'][initializer]
            self.state_normalization = state_normalization
            self.batch_normalization = batch_normalization
            self.state_space_samples = state_space_samples
            self.nodes_per_dense_layer = 100
            self.critic_weight_dict = {}

            # normalisation
            self.critic_weight_dict["state_norm"] = layers.Normalization(
                axis=-1)
            self.critic_weight_dict["state_norm"].adapt(
                self.state_space_samples)
            self.critic_weight_dict["batch_norm"] = layers.BatchNormalization()
            # common layers
            self.critic_weight_dict["dense1"] = layers.Dense(
                self.nodes_per_dense_layer, activation=self.activation_function, kernel_initializer=self.initializer)
            self.critic_weight_dict["dense2"] = layers.Dense(
                self.nodes_per_dense_layer, activation=self.activation_function, kernel_initializer=self.initializer)
            # outlet
            self.critic_weight_dict["value"] = layers.Dense(1)

        def call(self, inputs):
            """
            passes state forward through NN; method-name 'call()' can not be changed, since tf.keras uses this overload to build NN
            output: V(t)
            """
            x = inputs
            if (self.state_normalization == True):
                x = self.critic_weight_dict["state_norm"](x)
            x = self.critic_weight_dict["dense1"](x)
            if (self.batch_normalization == True):
                x = self.critic_weight_dict["batch_norm"](x)
            x = self.critic_weight_dict["dense2"](x)
            value = self.critic_weight_dict["value"](x)
            return value

        def critic_loss(self, discounted_rewards, predicted_values):
            """
            simple MSE between predicted V(t) and actual V(t)= discounted rewards
            """
            loss = keras.losses.mean_squared_error(
                discounted_rewards, predicted_values)
            return loss
