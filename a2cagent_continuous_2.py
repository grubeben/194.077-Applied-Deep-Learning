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


class A2CAgent(keras.Model):

    """__CONSTRUCTOR__"""

    def __init__(self, s0, act_space_high, act_space_low, state_space_samples, obs_dim, state_normalization, batch_normalization, use_existing_policy, add_branch_layer):
        super().__init__()
        """conveniece"""
        self.my_path = os.getcwd()
        self.model_path = 0

        """parameters"""
        self.critic_loss_weight = 0.4
        self.actor_loss_weight = 1
        self.entropy_loss_weight = 0.04
        self.batch_size = 64
        self.render_size= 10 #render every 10 steps
        self.gamma = 0.95
        self.obs_dim = obs_dim
        self.state_space_samples = state_space_samples

        """state"""
        self.s0 = s0

        """action space"""
        self.action_space_low = act_space_high
        self.action_space_high = act_space_low

        """ NN"""
        self.add_branch_layer = add_branch_layer
        self.batch_normalization = batch_normalization
        self.state_normalization = state_normalization
        self.nodes_per_dense_layer = 64
        self.my_weight_dict = {}

        # state normalisation
        self.my_weight_dict["state_norm"] = layers.Normalization(axis=-1)
        self.my_weight_dict["state_norm"].adapt(self.state_space_samples)
        # common layers
        self.my_weight_dict["dense1"] = layers.Dense(
            self.nodes_per_dense_layer, activation=tfa.activations.mish, kernel_initializer=keras.initializers.he_normal())
        self.my_weight_dict["batch_norm1"] = layers.BatchNormalization()
        self.my_weight_dict["dense2"] = layers.Dense(
            self.nodes_per_dense_layer, activation=tfa.activations.mish, kernel_initializer=keras.initializers.he_normal())
        self.my_weight_dict["batch_norm2"] = layers.BatchNormalization()
        # critic layers
        self.my_weight_dict["value_inbetween1"] = layers.Dense(
            self.nodes_per_dense_layer/2, activation=tfa.activations.mish, kernel_initializer=keras.initializers.he_normal())
        self.my_weight_dict["batch_norm_value"] = layers.BatchNormalization()
        self.my_weight_dict["value"] = layers.Dense(1)  # outlet 1
        # actor layers
        self.my_weight_dict["norm_inbetween1"] = layers.Dense(
            self.nodes_per_dense_layer/2, activation=tfa.activations.mish, kernel_initializer=keras.initializers.he_normal())
        self.my_weight_dict["batch_norm_norm"] = layers.BatchNormalization()
        self.my_weight_dict["mu"] = layers.Dense(1)  # outlet 2
        self.my_weight_dict["sigma_inbetween1"] = layers.Dense(1)
        self.my_weight_dict["sigma"] = layers.Dense(
            1, activation=tf.nn.softplus)  # outlet 3 | sigma has to be positive

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

    """losses"""

    def critic_loss(self, discounted_rewards, predicted_values):
        loss = keras.losses.mean_squared_error(
            discounted_rewards, predicted_values) * self.critic_loss_weight
        return loss

    # let's penalize large sigma? ==> better: lets penalize high uncertainty (=wide streched norm dist)
    def entropy_loss(self, norm_dist):
        return - norm_dist.entropy()

    def actor_loss(self, combined, norm_dist):
        actions = combined[:, 0]  # first column holds a_t
        advantages = combined[:, 1]  # second column holds A_t
        loss = -norm_dist.log_prob(actions)*advantages

        return loss * self.actor_loss_weight + self.entropy_loss_weight * self.entropy_loss(norm_dist)

    """NN methods"""

    # pass state forward through NN
    def call(self, inputs):
        # common part
        x = inputs
        if (self.state_normalization == True):
            x = self.my_weight_dict["state_norm"](x)
        x = self.my_weight_dict["dense1"](x)
        if (self.batch_normalization == True):
            x = self.my_weight_dict["batch_norm1"](x)
        x = self.my_weight_dict["dense2"](x)
        if (self.batch_normalization == True):
            x = self.my_weight_dict["batch_norm2"](x)

        # ciritc branch
        value = x
        if (self.add_branch_layer == True):
            value = self.my_weight_dict["value_inbetween1"](value)
        value = self.my_weight_dict["value"](value)
        if (self.batch_normalization == True):
            value = self.my_weight_dict["batch_norm_value"](value)

        # norm dist branch
        norm = x
        if (self.add_branch_layer == True):
            norm = self.my_weight_dict["norm_inbetween1"](x)
        if (self.batch_normalization == True):
            norm = self.my_weight_dict["batch_norm_norm"](norm)
        mu = self.my_weight_dict["mu"](norm)
        sigma = self.my_weight_dict["sigma_inbetween1"](norm)
        sigma = self.my_weight_dict["sigma"](sigma)

        # https://www.tensorflow.org/probability/examples/TensorFlow_Distributions_Tutorial
        norm_dist = tfp.distributions.Normal(mu, sigma)
        return value, norm_dist  # shallow structure

    def action_value(self, state):  # pass state through NN and choose action
        value, norm_dist = self.predict_on_batch(
            state)  # runs call() from above
        # sample from prob distribution and remove batch dimension
        action = tf.squeeze(norm_dist.sample(1), axis=0)

        # action = tf.clip_by_value(action, -10,10) #fit into action space for CartPoleContinuousBulletEnv
        # fit into action space MountainCar gym-CartPole
        action = tf.clip_by_value(action, -1, 1)
        # action = tf.clip_by_value(action, -2,2) #fit into action space  PENDULUM

        return action, value


# IMPROVEMENTS
# next: if environment has continuous observation space that can assume values [-inf,inf] (mountaincar) I should introduce state normalisation ==>
    # learning: from NO convergence without state normalisation to pretty good convergence
# next: load cartpole cont from git (to check whether bullet one has a problem) ==> no change
# next: add negative reward for episode end ==> no change ==> large loss in critic network, little loss in actor networ
# next: additional critic layer ==> a tiny bit better (seems to build up knowledge only to collapse after)
# next: maybe overparametrisation, try smaller network (maybe a bit better, but might be random)
# next: change learning rates? since i only use one NN i cannot do that individually (but this should be dealt with by the loss_weights?)
# next: decrease general learning rate ==> nope
# next: batch normalisation?

# next: pretrain value network?

# TODO
# add normalization etc to discrete, make graphs for diffrerent versions for report
# set up basic tests (in train loop?)
# make continuous work somehow
# read the docs?
# test function!!!
# visualization for test policy
