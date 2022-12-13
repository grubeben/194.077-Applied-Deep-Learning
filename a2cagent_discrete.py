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

        """env and update parameters"""
        self.critic_loss_weight = 0.4
        self.actor_loss_weight = 1
        self.entropy_loss_weight = 0.04
        self.batch_size = 64
        self.gamma = 0.95
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        """state"""
        self.s0 = s0

        """NN"""
        # network configuration
        self.metadata = {
            'activation.functions':     {'relu': keras.activations.relu, 'mish': tfa.activations.mish},
            'initialization.functions': {'normal': keras.initializers.he_normal(), 'xavier': keras.initializers.glorot_normal()}
        }

        self.activation_function = self.metadata['activation.functions'][activation_function]
        self.initializer = self.metadata['initialization.functions'][initializer]
        self.state_normalization = state_normalization
        self.state_space_samples = state_space_samples
        self.batch_normalization = batch_normalization

        self.my_weight_dict = {}

        # layers
        # state normalisation layer
        self.my_weight_dict["state_norm"] = layers.Normalization(axis=-1)
        self.my_weight_dict["state_norm"].adapt(self.state_space_samples)

        # main and batch-normalization layers
        self.my_weight_dict["dense1"] = layers.Dense(
            64, activation=self.activation_function, kernel_initializer=self.initializer)
        self.my_weight_dict["batch_norm"] = layers.BatchNormalization()
        self.my_weight_dict["dense2"] = layers.Dense(
            64, activation=self.activation_function, kernel_initializer=self.initializer)
        self.my_weight_dict["value"] = layers.Dense(1)
        self.my_weight_dict["policy_logits"] = layers.Dense(self.act_dim)

    """__METHODS__"""

    def discounted_rewards_advantages(self, rewards, V_batch, V_nplus1, dones=0):
        """
        computes discounted rewards and advantages from rewards collected, NN-value-function estimates (V(0-t)) during batch 
        and NN-value-function estimate (V(t+1)) for the step ahead

        """
        discounted_rewards = np.array(rewards + [V_nplus1])
        # go through V(t) in reverse order and discount
        for t in reversed(range(len(rewards))):
            discounted_rewards[t] = rewards[t] + self.gamma * \
                discounted_rewards[t+1] * (1-dones[t])
        discounted_rewards = discounted_rewards[:-1]
        # advantages are bootstrapped discounted rewards - values, using Bellman's equation
        advantages = discounted_rewards - np.stack(V_batch)
        return discounted_rewards, advantages

    """losses"""

    def critic_loss(self, discounted_rewards, predicted_values):
        """
        simple MSE between predicted V(t) and actual V(t)= discounted rewards
        """
        return keras.losses.mean_squared_error(discounted_rewards, predicted_values) * self.critic_loss_weight

    def entropy_loss(self, policy_logits):
        """
        we want the log-probabilities to be as unambiguous as possible
        """
        return -(keras.losses.categorical_crossentropy(policy_logits, policy_logits, from_logits=True) * self.entropy_loss_weight)

    def actor_loss(self, combined, policy_logits):
        """
        computes actor loss based on trace vectors from Session.train()

        [NOTE:  actions are mapped onto [0,1,2,..,act_dim]
                CategoricalCrossentropy maps the log-probabilities onto the same space before compuing the Crossentropy
                define sparse_ce object with sum-reduction (since we want the loss to be the some of step-losses)
                and set 'advantages' as sample weights]
        """
        # first column holds actions taken during batch
        actions = combined[:, 0]
        # second column holds advantages obtained during batch
        advantages = combined[:, 1]
        sparse_ce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        actions = tf.cast(actions, tf.int32)
        policy_loss = sparse_ce(actions, policy_logits,
                                sample_weight=advantages)

        return policy_loss * self.actor_loss_weight + self.entropy_loss(policy_logits) * self.entropy_loss_weight

    """NN methods"""

    def call(self, inputs):
        """
        passes state forward through NN; method-name 'call()' can not be changed, since tf.keras uses this overload to build NN

        output: V(t), normal_distribution(a)
        """
        
        x = inputs
        if (self.state_normalization == True):
            x = self.my_weight_dict["state_norm"](x)
        x = self.my_weight_dict["dense1"](x)
        if (self.batch_normalization == True):
            x = self.my_weight_dict["batch_norm"](x)
        x = self.my_weight_dict["dense2"](x)

        return self.my_weight_dict["value"](x), self.my_weight_dict["policy_logits"](x)

    def action_value(self, state):
        """
        choose action from normal_distribution(a)
        """
        value, logits = self.predict_on_batch(state)  # runs call() from above
        # choose random action based on action probabilities
        action = tf.random.categorical(logits, 1)[0]
        return action, value
