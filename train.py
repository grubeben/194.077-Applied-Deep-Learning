import gym
import pybullet as p
import pybullet_envs
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

import a2cagent

def train():
    #load environment
    env = gym.make("HopperBulletEnv-v0")
    _,state0=env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    #initiate agent
    model=a2cagent.A2CAgent(state0, act_dim, obs_dim, use_existing_policy=False, use_latest_policy=False)
    #compile NN with using inherited keras-method
    model.compile(optimizer=keras.optimizers.Adam(), loss=[model.critic_loss, model.actor_loss, model.entropy_loss])

    #set up TensorBoard to visualize progress
    train_writer = tf.summary.create_file_writer(model.my_path + f"/A2C{datetime.datetime.now().strftime('%d%m%Y%H%M')}")

    #main training loop





if __name__=="__main__":
    train()