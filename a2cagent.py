import os.path

import gym
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras import layers


class A2CAgent(keras.Model):

    """__CONSTRUCTOR__"""

    def __init__(self, s0, act_dim, obs_dim, state_space_samples, state_normalization, batch_normalization, activation_function, initializer, use_existing_policy):
        super().__init__()

        """conveniece"""
        self.my_path = os.getcwd()
        self.model_path = 0
        self.render_size = 10

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
        self.metadata = {
            'activation.functions': {'relu': keras.activations.relu, 'mish': tfa.activations.mish},
            'initialization.functions': {'normal': keras.initializers.he_normal(), 'xavier': keras.initializers.glorot_normal()}
        }

        self.activation_function = self.metadata['activation.functions'][activation_function]
        self.initializer = self.metadata['initialization.functions'][initializer]
        self.state_normalization = state_normalization
        self.state_space_samples = state_space_samples
        self.batch_normalization = batch_normalization

        self.my_weight_dict = {}

        # state normalisation
        self.my_weight_dict["state_norm"] = layers.Normalization(axis=-1)
        self.my_weight_dict["state_norm"].adapt(self.state_space_samples)

        # main layers
        self.my_weight_dict["dense1"] = layers.Dense(
            64, activation=self.activation_function, kernel_initializer=self.initializer)
        self.my_weight_dict["batch_norm"] = layers.BatchNormalization()
        self.my_weight_dict["dense2"] = layers.Dense(
            64, activation=self.activation_function, kernel_initializer=self.initializer)
        self.my_weight_dict["value"] = layers.Dense(1)
        self.my_weight_dict["policy_logits"] = layers.Dense(self.act_dim)

    """__METHODS__"""

    """reward and advantages"""

    def discounted_rewards_advantages(self, rewards, V_batch, V_nplus1, dones=0):
        discounted_rewards = np.array(rewards + [V_nplus1])
        for t in reversed(range(len(rewards))):
            discounted_rewards[t] = rewards[t] + self.gamma * \
                discounted_rewards[t+1] * (1-dones[t])
        discounted_rewards = discounted_rewards[:-1]
        # advantages are bootstrapped discounted rewards - values, using Bellman's equation
        advantages = discounted_rewards - np.stack(V_batch)
        return discounted_rewards, advantages

    """losses"""

    def critic_loss(self, discounted_rewards, predicted_values):
        return keras.losses.mean_squared_error(discounted_rewards, predicted_values) * self.critic_loss_weight

    def entropy_loss(self, policy_logits):
        return -(keras.losses.categorical_crossentropy(policy_logits, policy_logits, from_logits=True) * self.entropy_loss_weight)

    def actor_loss(self, combined, policy_logits):
        actions = combined[:, 0]  # first column holds a_t
        advantages = combined[:, 1]  # second column holds A_t
        sparse_ce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        actions = tf.cast(actions, tf.int32)
        policy_loss = sparse_ce(actions, policy_logits,
                                sample_weight=advantages)

        return policy_loss * self.actor_loss_weight + self.entropy_loss(policy_logits) * self.entropy_loss_weight

    """NN methods"""

    def call(self, inputs):  # pass state forward through NN
        x = inputs
        if (self.state_normalization == True):
            x = self.my_weight_dict["state_norm"](x)
        x = self.my_weight_dict["dense1"](x)
        if (self.batch_normalization == True):
            x = self.my_weight_dict["batch_norm"](x)
        x = self.my_weight_dict["dense2"](x)

        return self.my_weight_dict["value"](x), self.my_weight_dict["policy_logits"](x)

    def action_value(self, state):  # pass state through NN and choose action
        value, logits = self.predict_on_batch(state)  # runs call() from above
        # choose random action based on action probabilities
        action = tf.random.categorical(logits, 1)[0]
        return action, value
