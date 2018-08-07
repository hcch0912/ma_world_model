
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
import os

def pp_simulate(model, env, arglist, seed=-1, max_len=-1):


  if arglist.train_mode:
    num_episode = arglist.simu_episode
  else:
    num_episode = 1   
  prey_reward_list = []
  predator_reward_list = []
  t_list = []

  max_episode_length = 1000

  penalize_turning = False

  if arglist.train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

  for episode in range(num_episode):
    for i in range(len(model)):
      model[i].reset()
    obs = env.reset()
    prey_total_reward = 0.0
    predator_total_reward = 0.0

    random_generated_int = np.random.randint(2**31-1)

    prey_filename = os.path.join(arglist.data_dir, "prey",str(random_generated_int)+".npz")
    predator_filename = os.path.join(arglist.data_dir, "predator", str(random_generated_int)+".npz")
    recording_mu = [[]]*5
    recording_logvar = [[]]*5
    recording_action = [[]]*5
    recording_reward = [[]] *5

    for t in range(max_episode_length):

      if arglist.render_mode:
        env.render("human")
      else:
        env.render('rgb_array')

      obs = [Image.fromarray(o) for o in obs]
      obs = [o.resize((64,64),Image.ANTIALIAS) for o in obs]
      obs = np.array([np.array(o) for o in obs])
      

      action_episode = []
      z0, mu0,  logvar0 = model[0].encode_obs(obs[0])
      action0 = model[0].get_action(z0)
      action_episode.append(action0)
      recording_mu[0].append(mu0)
      recording_logvar[0].append(logvar0)
      recording_action[0].append(action0)
      
      for i in range(1,5):
        z1, mu1, logvar1 = model[1].encode_obs(obs[i])
        action1 = model[1].get_action(z1)
        action_episode.append(action1)
        recording_mu[i].append(mu1)
        recording_logvar[i].append( logvar1)
        recording_action[i].append(action1)

      #Supervise the tau training
      if arglist.supervise:
        for i in range(4):
          model[0].oppo_model.supervise_tau(action_episode[i+1], model[0].act_traj[i])
          for j in range(4):
            if j ==i: continue
            else:
              model[i].oppo_model.supervise_tau(action_episode[j+1],model[i+1].act_traj[j] )

        #update act_trajc
        for i in range(4):
          model[0].act_traj[i].append(action_episode[i])
          model[i+1].act_traj[0].append(action0)
          for j in range(4):
            if j == i: continue
            else:
              model[1+i].act_traj[j].append(action_episode[1+j])   

      obs, rewards, done, _ = env.step(action_episode)

      extra_reward = 0.0 # penalize for turning too frequently
      reward = 0.
      
      if arglist.train_mode and penalize_turning:
          extra_reward -= np.abs(action0[0])/10.0
          rewards[0] += extra_reward
          extra_reward -= np.abs(action1[0]) /10.0
          rewards[4] += extra_reward 
      # all predator set to have same rewards just for now #TODO to shape reward    
      predator_reward = rewards[4]   
      prey_reward = rewards[0]
      

      recording_reward[0].append(prey_reward)
      for i in range(1,5):
        recording_reward[i].append(predator_reward)

      prey_total_reward += prey_reward
      predator_total_reward += predator_reward  
      if done:
        break
     
  #TODO: modify here 
    #for recording:
    obs = [Image.fromarray(o) for o in obs]
    obs = [o.resize((64,64),Image.ANTIALIAS) for o in obs]
    obs = np.array([np.array(o) for o in obs])

    z0, mu0,  logvar0 = model[0].encode_obs(obs[0])
    action0 = model[0].get_action(z0)

    z0, mu0,  logvar0 = model[0].encode_obs(obs[0])
    action0 = model[0].get_action(z0)
    recording_mu[0].append(mu0)
    recording_logvar[0].append(logvar0)
    recording_action[0].append(action0)
      
    for i in range(1,5):
      z1, mu1, logvar1 = model[1].encode_obs(obs[i])
      action1 = model[1].get_action(z1)
      recording_mu[i].append(mu1)
      recording_logvar[i].append(logvar1)
      recording_action[i].append(action1)

    recording_mu = np.array(recording_mu, dtype=np.float16)
    recording_logvar = np.array(recording_logvar, dtype=np.float16)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)

    if not arglist.render_mode:
      if arglist.recording_mode:
        np.savez_compressed(prey_filename, mu=recording_mu[0], logvar=recording_logvar[0], action=recording_action[0], reward=recording_reward[0])
        for i in range(1,5):
          np.savez_compressed(predator_filename,  mu=recording_mu[i], logvar=recording_logvar[i], action=recording_action[i], reward=recording_reward[i])

    if arglist.render_mode:
      print(" prey total reward", prey_total_reward, "predator total reward",predator_total_reward ,"timesteps", t)
    prey_reward_list.append(prey_total_reward)
    predator_reward_list.append(predator_total_reward)
    t_list.append(t)

  return prey_reward_list, predator_reward_list , t_list
