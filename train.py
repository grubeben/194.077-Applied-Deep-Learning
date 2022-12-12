import a2cagent
import gym
import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras

from datetime import datetime
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import display
os.environ["SDL_VIDEODRIVER"] = "dummy"


class Session():
    """
    Defines a training session prior to starting training.

    env_str:                ['CartPole-v1'(act_dim=2) 'MountainCar-v0' (act_dim=3)] and any other gym environment with discrete action space
                            [NOTE: need to specify 'Session.act_dim' manually for some]
    converged_reward_limit: If mean reward over 100 episodes is >= 'converged_reward_limit' training stops
                            [NOTE: only well defined for CartPole-v1, for every other environment set to a very high number and adjust with 'train(max_num_batches)' only]
    activation_function:    ['relu', 'mish']
    initializer:            ['normal', 'xavier']
    state_normalization:    True/False
    batch_normalization:    True/False
    specification:          string (You can add a custom add_on to the name of the session)
    use_existing_policy:    True/False
    policy:                 =None / path to dir of form 'A2C281120221423CartPole-v1_mish' relative to AppliedDeep../training_discrete/ as string
                            [NOTE: if policy is not provided, you will be asked to provide the Session-name of the policy you want to reuse 
                            (path to dir of form 'A2C281120221423CartPole-v1_mish' relative to AppliedDeep../training_discrete/ dir as string)]
    """

    def __init__(self, converged_reward_limit=195, env_str="CartPole-v1", activation_function="relu", initializer="normal", state_normalization=False, batch_normalization=False, use_existing_policy=False, policy=None, specification=""):
        # session parameter
        self.converged_reward_limit = converged_reward_limit

        # load environment
        self.env = gym.make(env_str)
        self.s0 = self.env.reset()

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space
        if (env_str == "CartPole-v1"):
            self.act_dim = 2  # has to be set manually for agent class to understand
        if (env_str == "Acrobot-v1"):
            self.act_dim = 3  # has to be set manually for agent class to understand

        # state normalisation
        self.state_space_samples = np.array(
            [self.env.observation_space.sample() for x in range(6400)])
        if (env_str == "CartPole-v1"):
            # sample-method does not yield reasonable results for CartPole, so we load some historical state data
            self.state_space_samples = np.loadtxt(
                os.getcwd()+"/obs-samples/norm_a.txt")

        # initiate agent
        self.model = a2cagent.A2CAgent(
            self.s0, self.act_dim, self.obs_dim, self.state_space_samples, state_normalization, batch_normalization, activation_function, initializer, use_existing_policy)
        # compile NN with inherited keras-method
        self.model.compile(optimizer=keras.optimizers.Adam(), loss=[
            self.model.critic_loss, self.model.actor_loss])

        # load weights from existing model
        if use_existing_policy is True:
            if policy is None:
                policy = input(
                    "Please insert path to dir of form 'A2C281120221423CartPole-v1_mish' relative to 'AppliedDeep../training_discrete/' as string:\n")
                self.model.train_on_batch(tf.stack(np.zeros((self.model.batch_size, self.model.obs_dim))), [
                    np.zeros((self.model.batch_size, 1)), np.zeros((self.model.batch_size, 2))])
                self.model.load_weights(
                    self.model.my_path+'/training_discrete/' + policy+"/model/")

                """non WSL version"""
                # Tk().withdraw()
                # filename = askopenfilename()
                # self.model.load_weights(filename)

            else:
                self.model.load_weights(
                    self.model.my_path+'/training_discrete/' + policy+"/model/")

        # set up ModelCheckpoint
        self.model.model_path = self.model.my_path+'/training_discrete' + \
            f"/A2C{datetime.now().strftime('%d%m%Y%H%M')}" + \
            env_str+'_'+activation_function+'_'+initializer+'_'+specification
        os.makedirs(self.model.model_path+'/model/')

        # set up TensorBoard to visualize progress
        self.train_writer = tf.summary.create_file_writer(
            self.model.model_path + '/tensorboard')

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
                # obtain action distibution
                _, policy_logits = self.model(s.reshape(1, -1))
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
                        self.model.save_weights(
                            self.model.model_path+'/model/', save_format="tf")

                    episode_reward_sum = 0
                    episode += 1
                else:
                    rewards.append(reward)

            _, next_value = self.model.action_value(s.reshape(1, -1))
            discounted_rewards, advantages = self.model.discounted_rewards_advantages(
                rewards, values, next_value[0][0], dones)  # compute input for NN update
            # bring actions chosen and advantages into shape that keras method can work with
            combined = np.zeros((len(actions), 2))
            combined[:, 0] = actions
            combined[:, 1] = advantages
            # cperform NN update with keras method and obtain loss
            # discounted_rewards are used for the critic update, combined=[a_t, A_t] for the actor
            loss = self.model.train_on_batch(
                tf.stack(states), [discounted_rewards, combined], return_dict=False)
            with self.train_writer.as_default():
                tf.summary.scalar('tot_loss', np.sum(loss), batch)

            # make convergence check every 10 batches and only if reward was above necessary average
            if (episode_reward_sum > self.converged_reward_limit):
                if (np.mean(episode_rewards) >= episode_reward_sum):
                    print("\nCONVERGED in ", batch, " batches / ",
                          batch*self.model.batch_size, " steps")
                    return 0

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
            _, policy_logits = self.model(s.reshape(1, -1))
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

    """example for running pretrained "CartPole-v1" until convergence"""
    # session = Session(converged_reward_limit=195, env_str="CartPole-v1", specification="no_normalization", activation_function='relu', initializer='normal', state_normalization=False,
    #                   batch_normalization=False, use_existing_policy=False)
    # session.train(max_num_batches=1000)

    """example for running pretrained "CartPole-v1" until convergence"""
    # session = Session(converged_reward_limit=195, env_str="CartPole-v1", specification="based_on_pretrained", activation_function='mish', initializer='xavier', state_normalization=True,
    #                   batch_normalization=False, use_existing_policy=True, policy='pretrained/CartPole-v1_mish_normal_STATE_NORMALIZATION/')
    # session.train(max_num_batches=1000)

    """
    example for running pretrained "Acrobat-v1" until convergence

    [NOTE: for some interesting reason this training session behaves a bit like the continuous ones: most of the time without any success, but now and then you get lucky and the agent learns]
    """
    # session = Session(converged_reward_limit=-100, env_str="Acrobot-v1", specification="PROOF", activation_function='mish', initializer='normal', state_normalization=False,
    #                   batch_normalization=False, use_existing_policy=False)
    # session.train(max_num_batches=1000)
