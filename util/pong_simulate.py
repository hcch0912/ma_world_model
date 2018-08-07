import numpy as np
import random

import json
import sys

from two_player.pong import PongGame
from prey_predator.env import PreyPredatorEnv
import time
import argparse
from PIL import Image
# from vae.vae import ConvVAE
# from rnn.rnn import hps_model, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size



def pong_simulate(mode, arglist, seed = -1, max_len = -1):

  reward_list = []
  t_list = []

  max_episode_length = 1000

  penalize_turning = False

  if train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

  for episode in range(num_episode):

    model.reset()

    obs = model.env.reset()
    # obs = Image.fromarray(obs)
    
    total_reward = 0.0

    random_generated_int = np.random.randint(2**31-1)
    
    # filename = arglist.data_dir +"/"+str(random_generated_int)+".npz"
    filename = arglist.data_dir+str(random_generated_int)+".npz"

    recording_mu = []
    recording_logvar = []
    recording_action = []
    recording_reward = [0]

    for t in range(max_episode_length):

      if render_mode:
        model.env.render("human")
      else:
        model.env.render('rgb_array')
      obs = Image.fromarray(obs)
      obs = obs.resize((64,64),Image.ANTIALIAS)
      obs = np.array(obs)
      z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(z)

      recording_mu.append(mu)
      recording_logvar.append(logvar)
      recording_action.append(action)
      recording_reward = []
      if arglist.competitive:
        obs, rewards, [act1, act2], goals, win = model.env.step([action[0], 'script'])
      else: 
        obs, rewards, [act1, act2], goals, win = model.env.step(action)

      extra_reward = 0.0 # penalize for turning too frequently
      reward = 0.
      if arglist.competitive:
        if train_mode and penalize_turning:
          extra_reward -= np.abs(action[0])/10.0
          rewards[0] += extra_reward
        reward = rewards[0]
      else:
        if train_mode and penalize_turning:
          reward = np.sum(rewards)
          extra_reward -= np.abs(action[0])/10.0
          reward += extra_reward

      recording_reward.append(reward)
      total_reward += reward  
      if win:
        break

    #for recording:
     # obs = Image.fromarray(obs)
    obs = Image.fromarray(obs)
    obs = obs.resize((64,64),Image.ANTIALIAS)
    z, mu, logvar = model.encode_obs(obs)
    action = model.get_action(z)
    recording_mu.append(mu)
    recording_logvar.append(logvar)
    recording_action.append(action)

    recording_mu = np.array(recording_mu, dtype=np.float16)
    recording_logvar = np.array(recording_logvar, dtype=np.float16)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)

    if not render_mode:
      if arglist.recording_mode:
       	np.savez_compressed(filename, mu=recording_mu, logvar=recording_logvar, action=recording_action, reward=recording_reward)

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list
