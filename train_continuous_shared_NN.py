import gym
import pybullet as p
import pybullet_envs
import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras

from datetime import datetime
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename


import a2cagent_continuous_shared_NN as a2cc
import continuous_cartpole_env as continuous_cartpole


def train(num_batches=10000, env_str="ContinuousCartPoleEnv", add_branch_layer=False, use_existing_policy=False, state_normalization=False, batch_normalization=False, specification=""):
    # check for custom CARTPOLE ENV
    custom_cart = False
    if (env_str == "ContinuousCartPoleEnv"):
        custom_cart = True

    # load environment
    env = None
    if (custom_cart == True):
        env = continuous_cartpole.ContinuousCartPoleEnv()
    else:
        env = gym.make(env_str)
    s0 = env.reset()

    act_space_low = env.action_space.low[0]
    act_space_high = env.action_space.high[0]
    obs_dim = env.observation_space.shape[0]

    # state normalisation
    state_space_samples = np.array(
        [env.observation_space.sample() for x in range(6400)])
    if (env_str == "CartPoleContinuousBulletEnv-v0" or custom_cart == True):
        # method does not yield reasonable results for CartPole, so we load some historical state data
        state_space_samples = np.loadtxt(os.getcwd()+"/obs-samples/norm_a.txt")

    # initiate agent
    model = a2cc.A2CAgent(s0, act_space_high, act_space_low, state_space_samples, obs_dim, state_normalization, batch_normalization,
                          use_existing_policy, add_branch_layer)
    # compile NN with inherited keras-method
    # add gradient clipping keras.optimizers.Adam(clipvalue=0.5) clipping keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=keras.optimizers.Adam(), loss=[
                  model.critic_loss, model.actor_loss])

    # load weights from existing model
    if use_existing_policy == True:
        # string in format 'A2C281120221423CartPole-v1_mish'
        policy = input(
            "insert directory corresponding to policy from which to start from:")
        model.train_on_batch(tf.stack(np.zeros((model.batch_size, model.obs_dim))), [
                             np.zeros((model.batch_size, 1)), np.zeros((model.batch_size, 2))])
        model.load_weights(
            model.my_path+'/training_continuous/models/' + policy+"/")

        """non WSL version"""
        # Tk().withdraw()
        # filename = askopenfilename()
        # model.load_weights(filename)

    # set up ModelCheckpoint
    model.model_path = model.my_path+'/training_continuous/models' + \
        f"/A2C{datetime.now().strftime('%d%m%Y%H%M')}" + \
        env_str+'_'+specification+'/'
    os.makedirs(model.model_path)

    # set up TensorBoard to visualize progress
    train_writer = tf.summary.create_file_writer(
        model.my_path + '/training_continuous/tensorboard'+f"/A2C{datetime.now().strftime('%d%m%Y%H%M')}"+env_str+"_"+specification)

    # main training loop
    episode_reward_sum = 0
    episode_reward_sum_best = 0
    s = env.reset()
    episode = 1
    loss = 0
    for step in range(num_batches):
        rewards = []
        actions = []
        values = []
        states = []
        dones = []
        for i in range(model.batch_size):
            # obtain action distibution
            _, _ = model(s.reshape(1, -1))
            a_t, V_t = model.action_value(s.reshape(1, -1))  # choose action
            s_new, reward, done, _ = env.step(a_t.numpy()[0])  # make step
            actions.append(a_t.numpy()[0])  # append trace vectors
            values.append(V_t[0][0])
            states.append(s)
            dones.append(done)
            episode_reward_sum += reward

            s = s_new

            # handle end of episode
            if done:
                rewards.append(0.0)
                s = env.reset()
                # loss will be 0 for first batch
                print(
                    f"Episode: {episode}, latest episode reward: {episode_reward_sum}, loss: {loss}")
                with train_writer.as_default():
                    tf.summary.scalar('rewards', episode_reward_sum, episode)

                # safe best model-version for later use
                if (episode_reward_sum >= episode_reward_sum_best):
                    episode_reward_sum_best = episode_reward_sum
                    model.save_weights(model.model_path, save_format="tf")

                episode_reward_sum = 0
                episode += 1
            else:
                rewards.append(reward)

        _, next_value = model.action_value(s.reshape(1, -1))
        discounted_rewards, advantages = model.discounted_rewards_advantages(
            rewards, values, next_value[0][0], dones)  # compute input for NN update

        # bring actions chosen and advantages into shape that keras method can work with
        combined = []
        for i in range(len(actions)):
            combined.append([actions[i][0], advantages[i]])

        combined = np.array(combined)

        loss = model.train_on_batch(
            tf.stack(states), [discounted_rewards, combined])

        with train_writer.as_default():
            tf.summary.scalar('tot_loss', np.sum(loss), step)


if __name__ == "__main__":

    # "CartPoleContinuousBulletEnv-v0", "ContinuousCartPoleEnv", "MountainCarContinuous-v0", "Pendulum-v1",  "HopperBulletEnv-v0", "(LunarLanderContinuous-v2")
    train(num_batches=2000, env_str="ContinuousCartPoleEnv",
          specification="confirm_prior_achievement", add_branch_layer=False, state_normalization=True, batch_normalization=False, use_existing_policy=False)
