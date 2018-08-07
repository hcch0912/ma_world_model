'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''
import os
import tensorflow as tf
import random
import numpy as np
import argparse
from vae.vae import ConvVAE, reset_graph
from PIL import Image
import time
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems

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
    parser.add_argument("--batch_size", type = int, default = 1000, help = "set the batch_size")
    parser.add_argument("--max_episode_len", type = int, default = 50, help = "max episode length")
    parser.add_argument("--warm_up_steps", type = int, default = 1000, help = "set the warm up steps")
    parser.add_argument("--lr", type = float, default = 0.0001, help = "learning rate")
    parser.add_argument("--gamma", type = float, default = 0.99, help = "discount rate")
    parser.add_argument("--kl_tolerance", type = float, default = 0.5, help = "dk divergence tolerance")
    parser.add_argument("--data_dir", type = str,default = "./record")
    parser.add_argument("--model_save_path", type = str,default = "./tf_vae", help= "model save path")
    parser.add_argument("--z_size", type = int, default = 32, help = "z size")
    parser.add_argument("--save_period", type = int, default = 30, help = "save every x timesteps")
    parser.add_argument("--use_image", type = bool, default = False, help = "use image or record")
    return parser.parse_args()

def count_length_of_filelist(filelist, data_dir):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join(data_dir, filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return  total_length

def create_dataset_with_image(filelist,data_dir, N=10000, M=1000): # N is 10000 episodes, M is number of timesteps
  # data = np.zeros((M*N, 64, 64, 3), dtype=np.uint8)
  dataset = []
  for i in range(len(filelist)):
    filename = filelist[i]
    img = Image.open(os.path.join(data_dir, filename))
    img = img.resize((64,64),Image.ANTIALIAS)
    dataset.append(np.array(img))
  return dataset


def create_dataset(filelist,arglist, N=10000, M=1000): # N is 10000 episodes, M is number of timesteps
  N = len(filelist)
  data = np.zeros((M*N, 64, 64, 3), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join(arglist.data_dir, filename))['obs']
    l = len(raw_data)
    if (idx+l) > (M*N):
      data = data[0:idx]
      print('premature break')
      break
    data[idx:idx+l] = raw_data
    idx += l
    if ((i+1) % 100 == 0):
      print("loading file", i+1)
  return data

if __name__ == '__main__':
    arglist = parse_args()

    if not os.path.exists(arglist.model_save_path):
      os.makedirs(arglist.model_save_path)

    log_path = './logs/{}_{}_{}.csv'.format(arglist.kl_tolerance, arglist.seed, time.time())
    log = open(log_path, '+w', 1)
    # load dataset from record/*. only use first 10K, sorted by filename.
    filelist = os.listdir(arglist.data_dir)
    filelist.sort()
    filelist = filelist[0:10000]
    #print("check total number of images:", count_length_of_filelist(filelist))
    if arglist.use_image:
      dataset = create_dataset_with_image(filelist, arglist.data_dir)
    else:  
      dataset = create_dataset(filelist, arglist)

    # split into batches:
    total_length = len(dataset)
    num_batches = int(np.floor(total_length/arglist.batch_size))
    print("num_batches", num_batches)

    reset_graph()

    vae = ConvVAE(z_size=arglist.z_size,
                  batch_size=arglist.batch_size,
                  learning_rate=arglist.lr,
                  kl_tolerance=arglist.kl_tolerance,
                  is_training=True,
                  reuse=False,
                  gpu_mode=True)

    # train loop:
    print("train", "step", "loss", "recon_loss", "kl_loss")
    for epoch in range(arglist.epoch):
      np.random.shuffle(dataset)
      for idx in range(num_batches):
        batch = dataset[idx*arglist.batch_size:(idx+1)*arglist.batch_size]

        obs = np.array(batch).astype(np.float)/255.0

        feed = {vae.x: obs,}

        (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
          vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
        ], feed)
      
        print("step", (train_step+1), train_loss, r_loss, kl_loss)
        log.write('{}\n'.format(','.join(map(str, 
                    [train_loss, r_loss, kl_loss, train_step]))))
        if ((train_step+1) % arglist.save_period == 0):
          vae.save_json(os.path.join(arglist.model_save_path,"vae.json"))

    # finished, final model:
    vae.save_json(os.path.join(arglist.model_save_path,"vae.json"))
