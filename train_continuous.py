import continuous_cartpole_env as continuous_cartpole
import a2cagent_continuous as a2cagent

from tkinter.filedialog import askopenfilename
from tkinter import Tk
import os
from datetime import datetime

from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pygame
import pybullet_envs
import pybullet as p
import gym

# for WSL rendering solution
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import display
os.environ["SDL_VIDEODRIVER"] = "dummy"


class Session():
    """
    Defines a training session prior to starting training.

    env_str:                ["ContinuousCartPoleEnv" "CartPoleContinuousBulletEnv-v0" "MountainCarContinuous-v0"]

    converged_reward_limit: If mean reward over 100 episodes is >= 'converged_reward_limit' training stops
                            [NOTE: only well defined for CartPole, for every other environment set to a very high number and adjust with 'train(max_num_batches)' only]
    specification:          string (You can add a custom add_on to the name of the session)
    use_existing_policy:    True/False
    policy:                 =None / path to dir of form 'A2C281120221423CartPole-v1_mish' relative to AppliedDeep../training_continuous/ as string
                            [NOTE: if policy is not provided, you will be asked to provide the Session-name of the policy you want to reuse 
                            (path to dir of form 'A2C281120221423CartPole-v1_mish' relative to AppliedDeep../training_continuous/ dir as string)]
    """

    def __init__(self, converged_reward_limit=195, env_str="ContinuousCartPoleEnv", use_existing_policy=False, policy=None, specification=""):
        # session parameter
        self.converged_reward_limit = converged_reward_limit

        # check for custom CARTPOLE ENV
        custom_cart = False
        if (env_str == "ContinuousCartPoleEnv"):
            custom_cart = True

        # load environment
        self.env = None
        if (custom_cart == True):
            self.env = continuous_cartpole.ContinuousCartPoleEnv()
        else:
            self.env = gym.make(env_str)
        self.s0 = self.env.reset()

        self.act_space_low = self.env.action_space.low[0]
        self.act_space_high = self.env.action_space.high[0]
        self.obs_dim = self.env.observation_space.shape[0]

        # load state normalization data
        self.state_space_samples = np.array(
            [self.env.observation_space.sample() for x in range(6400)])
        if (env_str == "CartPoleContinuousBulletEnv-v0" or custom_cart == True):
            # method does not yield reasonable results for CartPole, so we load some historical state data
            self.state_space_samples = np.loadtxt(
                os.getcwd()+"/obs-samples/norm_a.txt")

        # initiate agent
        self.model = a2cagent.agent(
            self.s0, self.act_space_high, self.act_space_low, self.obs_dim, self.state_space_samples)

        # set learning rates
        lr_actor = 0.00002
        lr_critic = 0.001

        # compile NN with inherited keras-method
        self.model.a.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_actor), loss=self.model.a.actor_loss)
        self.model.c.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_critic), loss=self.model.c.critic_loss)

        # load weights from existing model
        if use_existing_policy is True:
            if policy is None:
                policy = input(
                    "Please insert path to dir of form 'A2C281120221423CartPole-v1_mish' relative to 'AppliedDeep../training_continuous/' as string:\n")
                self.model.actor.train_on_batch(tf.stack(np.zeros(
                    (self.model.batch_size, self.model.obs_dim))), np.zeros((self.model.batch_size, 2)))
                self.model.critic.train_on_batch(tf.stack(np.zeros(
                    (self.model.batch_size, self.model.obs_dim))), np.zeros((self.model.batch_size, 1)))

                self.model.a.load_weights(
                    self.model.my_path+'/training_continuous/' + policy+"/modelc/actor/")
                self.model.c.load_weights(
                    self.model.my_path+'/training_continuous/' + policy+"/modelc/critic/")

            else:
                self.model.a.train_on_batch(tf.stack(np.zeros(
                    (self.model.batch_size, self.model.obs_dim))), np.zeros((self.model.batch_size, 2)))
                self.model.c.train_on_batch(tf.stack(np.zeros(
                    (self.model.batch_size, self.model.obs_dim))), np.zeros((self.model.batch_size, 1)))

                self.model.a.load_weights(
                    self.model.my_path+'/training_continuous/' + policy+"/modelc/actor/")
                self.model.c.load_weights(
                    self.model.my_path+'/training_continuous/' + policy+"/modelc/critic/")

        # set up ModelCheckpoints
        path_base = self.model.my_path+'/training_continuous' + \
            f"/A2C{datetime.now().strftime('%d%m%Y%H%M')}" + \
            env_str+'_mish_xavier_'+specification

        self.model.a.model_path = path_base + '/model/actor/'
        self.model.c.model_path = path_base + '/model/critic/'

        os.makedirs(self.model.a.model_path)
        os.makedirs(self.model.c.model_path)

        # set up TensorBoard to visualize progress
        self.train_writer = tf.summary.create_file_writer(
            path_base + '/tensorboard')

    def train(self, max_num_batches):
        """ 
        main training loop

        max_num_batches: number of batches after which training stops even if convergence is not reached

        """

        episode_rewards = np.empty(100)
        episode_reward_sum = 0
        episode_reward_sum_best = -500
        s = self.env.reset()
        episode = 1
        batch = 0
        loss = 0

        for batch in range(max_num_batches):
            rewards = []
            actions = []
            values = []
            states = []
            dones = []
            for i in range(self.model.batch_size):
                # run once for initiation
                _ = self.model.a(s.reshape(1, -1))
                _ = self.model.c(s.reshape(1, -1))
                a_t, V_t = self.model.action_value(
                    s.reshape(1, -1))  # choose action
                s_new, reward, done, _ = self.env.step(
                    a_t.numpy()[0])  # make step
                actions.append(a_t.numpy()[0])  # append trace vectors
                values.append(V_t[0][0])
                states.append(s)
                dones.append(done)
                episode_reward_sum += reward

                s = s_new

                # handle end of episode
                if done:
                    rewards.append(0.0)
                    s = self.env.reset()
                    # loss will be 0 for first batch
                    print(
                        f"Episode: {episode}, latest episode reward: {episode_reward_sum}, loss: {loss}")
                    with self.train_writer.as_default():
                        tf.summary.scalar(
                            'rewards', episode_reward_sum, episode)
                    # check for convergence of sufficient quality
                    episode_rewards = np.append(
                        episode_rewards, episode_reward_sum)
                    episode_rewards = np.delete(episode_rewards, 0)

                    # safe best model-version for later use
                    if (episode_reward_sum >= episode_reward_sum_best):
                        episode_reward_sum_best = episode_reward_sum
                        self.model.a.save_weights(
                            self.model.a.model_path, save_format="tf")
                        self.model.c.save_weights(
                            self.model.c.model_path, save_format="tf")

                    episode_reward_sum = 0
                    episode += 1
                else:
                    rewards.append(reward)

            _, next_value = self.model.action_value(s.reshape(1, -1))
            discounted_rewards, advantages = self.model.discounted_rewards_advantages(
                rewards, values, next_value[0][0], dones)  # compute input for NN update

            # bring actions chosen and advantages into shape that keras method can work with
            combined = []
            for i in range(len(actions)):
                combined.append([actions[i][0], advantages[i]])

            combined = np.array(combined)

            # cperform NN update with keras method and obtain loss
            loss_c = self.model.c.train_on_batch(
                tf.stack(states), discounted_rewards)
            loss_a = self.model.a.train_on_batch(tf.stack(states), combined)

            #combine losses
            loss=[loss_c, loss_a]

            #log them
            with self.train_writer.as_default():
                tf.summary.scalar('tot_loss', np.sum(loss), batch)

            # make convergence check only if reward was above necessary average (checking after every episode is costly)
            if (episode_reward_sum > self.converged_reward_limit):
                if (np.mean(episode_rewards) >= episode_reward_sum):
                    print("\nCONVERGED in ", batch, " batches / ",
                          batch*self.model.batch_size, " steps")
                    return 0

        # max_num_batches reached without convergence
        print("\nNOT CONVERGED in ", batch, " batches / ",
              batch*self.model.batch_size, " steps")

    def test(self, num_episodes):
        """
        Runs the current best policy learned during the Session() and renders it
        If you want to visualize a policy from a different session, initialize a new session object with 'use_existing_policy=True argument'
        [NOTE: IPyhton Displays are not available on WSL, this rendering method is a workaround
               Should only be called by a jupyter notebook!]

        num_episodes: The number of episodes that will be visualized
        """
        episode_reward_sum = 0
        s = self.env.reset()
        episode = 1
        step = 0
        while (episode <= num_episodes):
            # obtain action distibution
            _ = self.model.actor(s.reshape(1, -1))
            _ = self.model.critic(s.reshape(1, -1))
            a_t, V_t = self.model.action_value(
                s.reshape(1, -1))  # choose action
            s_new, reward, done, _ = self.env.step(a_t.numpy()[0])  # make step

            # render1 - in jupyter notebook
            if (step % (self.model.render_size) == 0):
                clear_output(wait=True)
                x = self.env.render(mode='rgb_array')
                plt.imshow(x)
                plt.show()

            episode_reward_sum += reward

            s = s_new

            # handle end of episode
            if done:
                s = self.env.reset()
                with self.train_writer.as_default():
                    tf.summary.scalar('rewards', episode_reward_sum, episode)

                episode_reward_sum = 0
                episode += 1


if __name__ == "__main__":

    # "CartPoleContinuousBulletEnv-v0", "ContinuousCartPoleEnv", "MountainCarContinuous-v0", "Pendulum-v1",  "HopperBulletEnv-v0", "(LunarLanderContinuous-v2")

    """example for trying to run "ContinuousCartPoleEnv" until convergence"""
    # session = Session(converged_reward_limit=195, env_str="ContinuousCartPoleEnv",
    #                   specification="TEST", use_existing_policy=False)
    # session.train(max_num_batches=1000)

    """example for trying to run "MountainCarContinuous-v0" until convergence"""
    session = Session(converged_reward_limit=90, env_str="MountainCarContinuous-v0",
                      specification="TEST", use_existing_policy=False)
    session.train(max_num_batches=1000)
