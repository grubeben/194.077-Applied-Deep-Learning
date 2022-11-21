import gym
import pybullet as p
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

import os
import os.path
import glob
import shutil
import json

import train


class A2CAgent(keras.Model):

    """__CONSTRUCTOR__"""

    def __init__(self,s0, act_dim, obs_dim, use_existing_policy=False, use_latest_policy=False):
        super().__init__()
        """conveniece"""
        self.my_path = os.getcwd()
        self.use_existing_policy = use_existing_policy
        self.use_latest_policy = use_latest_policy

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

        self.dense1 = keras.layers.Dense(
            64, activation='relu', kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(
            64, activation='relu', kernel_initializer=keras.initializers.he_normal())
        # Instead of creating two entire NNs we just use one with two outlet channels 
        # only the last layer weights decide whether we obtain P(s,a) or V(s)
        self.value = keras.layers.Dense(1)  # outlet 1
        self.policy_logits = keras.layers.Dense(self.act_dim)  # outlet 2
        self.probs = tf.nn.softmax(self.policy_logits)

    """__METHODS__"""

    """reward and advantages"""
    def discounted_rewards_advantages(self,rewards,V_batch,V_nplus1,dones=0):
        if dones==0:dones=np.zeros(self.batch_size)
        dsicounted_rewards=np.array([rewards(0)+self.gamma*V_nplus1])
        for i in range(len(rewards)-1):
            #done[i]= 1 if episode stopped during step i, then discounted prior rewards are set to zero
            discounted_rewards+=(rewards(i+1)+np.sum(discounted_rewards)*self.gamma*(1-dones[i])+dones[i]*self.gamma*V_nplus1)

        advantages= np.subtract(discounted_rewards,V_batch)
        return discounted_rewards, advantages
    
    """losses"""
    def critic_loss(self,discounted_rewards, predicted_values):
        return keras.losses.mean_squared_error(discounted_rewards, predicted_values) * self.critic_loss_weight

    def entropy_loss(self,probs):
        return -keras.losses.categorical_crossentropy(probs, probs) * self.entropy_loss_weight

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
        return policy_loss * self.actor_loss_weight

    """NN methods"""
    def call(self, inputs):  # pass state forward through NN
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.value(x), self.policy_logits(x)

    def action_value(self, state):  # pass state through NN and choose action
        value, logits = self.predict_on_batch(state)  # runs call() from above
        action = tf.random.categorical(logits, 1)[0] # choose random action based on action probabilities
        return action, value

    """start from presaved policy?"""
    # if self.use_existing_policy == True:
    #     if self.use_latest_policy == True:
    #         self.load_policy()
    #     else:
    #         self.load_policy(latest_simulation=False)
    #     self.define_s()

    # else:
    #     self.define_s()


def run_episode(env, policy, scaler, visualize=False):
    """Run single episode with option to visualize.
    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        visualize: boolean, True uses env.render() method to visualize episode
    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if visualize:
            env.render()
        obs = np.concatenate([obs, [step]])  # add time step feature
        obs = obs.astype(np.float32).reshape((1, -1))
        unscaled_obs.append(obs)
        # center and scale observations
        obs = np.float32((obs - offset) * scale)
        observes.append(obs)
        action = policy.sample(obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action.flatten())
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float32), np.concatenate(unscaled_obs))


"""_adapt default reward_"""

"""___CONVENIENCE METHODS___"""


# def store_policy(self):
#     my_path = os.getcwd()
#     date = str(datetime.now().date()).strip().replace("-", "_")
#     time = str(datetime.now().time()).strip().replace(":", "_")
#     time = time[0:-9]
#     self.total_path = (my_path + '\policies\\' +
#                        date + '_' + time)
#     os.mkdir(self.total_path)
#     actor_weights_path = (self.total_path + '\\'+'actor_weights' + '.npy')
#     critic_weights_path = (self.total_path + '\\' +
#                            'critic_weights' + '.npy')
#     git_path = (self.total_path + '\\' + 'git_head' + '.txt')
#     train_path = (self.total_path + '\\' + 'trainlog' + '.txt')
#     par_path = (self.total_path + '\\'+'parameterlog' + '.txt')

#     original = my_path[0:-5]+"\.git\FETCH_HEAD"
#     shutil.copyfile(original, git_path)

#     np.save(actor_weights_path, self.actor_weights)
#     np.save(critic_weights_path, self.critic_weights)
#     with open(train_path, "w") as file:
#         file.write(json.dumps(self.train_log))
#     with open(par_path, "w") as file:
#         file.write(json.dumps(self.parameter))

#     print(f"policy was saved in {self.total_path}")
