'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
from vae.vae import ConvVAE, reset_graph
import re
import argparse
from PIL import Image
import copy

os.environ["CUDA_VISIBLE_DEVICES"]="0"



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for stochastic games")
    parser.add_argument("--agent", type =str, default = 'DDPG', help = "agent type")
    parser.add_argument("--adv_agent", type = str, default = "script", help = "adv agent type")
    parser.add_argument("--game", type=str, default="Pong-2p-v0", help="name of the  env")
    parser.add_argument("--timestep", type= int, default= 2, help ="the time step of act_trajectory")
    parser.add_argument("--iteration", type=int, default=60000, help="number of episodes")
    parser.add_argument("--seed", type = int, default = 10, help = "random seed")
    parser.add_argument("--epoch", type = int, default = 10, help = "training epoch")
    parser.add_argument("--episodes", type = int, default = 100, help = "episodes")
    parser.add_argument("--steps", type = int, default = 50, help ="steps in one episode" )
    parser.add_argument("--batch_size", type = int, default = 5000, help = "set the batch_size")
    parser.add_argument("--max_episode_len", type = int, default = 50, help = "max episode length")
    parser.add_argument("--warm_up_steps", type = int, default = 1000, help = "set the warm up steps")
    parser.add_argument("--lr", type = float, default = 0.0001, help = "learning rate")
    parser.add_argument("--gamma", type = float, default = 0.99, help = "discount rate")
    parser.add_argument("--kl_tolerance", type = float, default = 0.5, help = "dk divergence tolerance")
    parser.add_argument("--data_dir", type = str,default = "./record")
    parser.add_argument("--series_dir", type = str, default = "./series")
    parser.add_argument("--vae_path", type = str,default = "./tf_vae", help= "model save path")
    parser.add_argument("--z_size", type = int, default = 32, help = "z size")
    parser.add_argument("--use_vae", type = bool, default = True, help = "use vae to get z or not")
    return parser.parse_args()


def load_raw_data_list(filelist,arglist):
  data_list = []
  action_list = []
  oppo_action_list = []
  counter = 0
  
  for i in range(len(filelist)):
    filename = filelist[i]
    raw_data = np.load(os.path.join(arglist.data_dir, filename))
    data_list.append(raw_data['obs'])
    action_list.append(raw_data['action'])
    oppo_action_list.append(raw_data['oppo_actions'])
    if ((i+1) % 1000 == 0):
      print("loading file", (i+1))
  
  return data_list, action_list, oppo_action_list

def encode_batch(batch_img,arglist):
  simple_obs = np.copy(batch_img).astype(np.float)/255.0
  simple_obs = simple_obs.reshape(arglist.batch_size, 64, 64, 3)
  mu, logvar = vae.encode_mu_logvar(simple_obs)
  z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))

  return mu, logvar, z

def decode_batch(batch_z,arglist):
  # decode the latent vector
  batch_img = vae.decode(z.reshape(arglist.batch_size, z_size)) * 255.
  batch_img = np.round(batch_img).astype(np.uint8)
  batch_img = batch_img.reshape(arglist.batch_size, 64, 64, 3)
  return batch_img




if __name__ == '__main__':
    arglist = parse_args()

    if not os.path.exists(arglist.series_dir):
        os.makedirs(arglist.series_dir)

    filelist = os.listdir(arglist.data_dir)
    filelist.sort()
    filelist = filelist[0:10000]

    dataset, action_dataset, oppo_action_dataset = load_raw_data_list(filelist, arglist)

    reset_graph()
    if arglist.use_vae:
      vae = ConvVAE(z_size=arglist.z_size,
                    batch_size=arglist.batch_size,
                    learning_rate=arglist.lr,
                    kl_tolerance=arglist.kl_tolerance,
                    is_training=False,
                    reuse=False,
                    gpu_mode=True) # use GPU on batchsize of 1000 -> much faster

      vae.load_json(os.path.join(arglist.vae_path, 'vae.json'))

      
    mu_dataset = []
    logvar_dataset = []
    for i in range(len(dataset)):
      data_batch = dataset[i]
      if len(data_batch) != arglist.batch_size:
            break
      if arglist.use_vae :      
        mu, logvar, z = encode_batch(data_batch, arglist)
        mu_dataset.append(mu.astype(np.float16))
        logvar_dataset.append(logvar.astype(np.float16))
      
      if ((i+1) % 100 == 0):
        print(i+1)
    action_dataset = np.array(action_dataset)
    oppo_action_dataset = np.array(oppo_action_dataset)    
    if arglist.use_vae:
      mu_dataset = np.array(mu_dataset)
      logvar_dataset = np.array(logvar_dataset)
      np.savez_compressed(os.path.join(arglist.series_dir, "series.npz"), action=action_dataset, oppo_action= oppo_action_dataset, mu=mu_dataset, logvar=logvar_dataset)

    else:  
      obs_dataset = np.array(dataset)  
      np.savez_compressed(os.path.join(arglist.series_dir, "series.npz"), action=action_dataset, oppo_action= oppo_action_dataset, obs = obs_dataset)

    


    