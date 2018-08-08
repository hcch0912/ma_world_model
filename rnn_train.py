'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
import time
from vae.vae import ConvVAE, reset_graph
from rnn.rnn import HyperParams, MDNRNN
import argparse
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for stochastic games")
    parser.add_argument("--agent", type =str, default = 'DDPG', help = "agent type")
    parser.add_argument("--adv_agent", type = str, default = "script", help = "adv agent type")
    parser.add_argument("--game", type=str, default="Pong-2p-v0", help="name of the  env")
    parser.add_argument("--timestep", type= int, default= 5, help ="the time step of act_trajectory")
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
    parser.add_argument("--model_save_path", type = str,default = "./tf_rnn/", help= "model save path")
    parser.add_argument("--z_size", type = int, default = 32, help = "z size")
    parser.add_argument("--initial_z_save_path", type = str, default = "tf_initial_z", help = "intial_z")
    parser.add_argument("--series_dir", type = str, default = "./series")
    parser.add_argument("--agent_num", type = int, default = 5, help = "agent number in the env")
    parser.add_argument("--action_space", type = int, default = 2, help = "action space size for each agent")
    parser.add_argument("--use_vae", type = bool, default = True, help ="use vae to get z")
    return parser.parse_args()



def random_batch(arglist):
  indices = np.random.permutation(len(data_action))[0:batch_size]
  action = data_action[indices]
  oppo_action = data_oppo_action[indices]
  if arglist.use_vae:
    mu = data_mu[indices]
    logvar = data_logvar[indices]
    s = logvar.shape
    z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
    return z, action, oppo_action
  else:
    obs = data_obs[indices]
    return obs, action, oppo_action  
  
 
  

def default_hps():
  return HyperParams(num_steps=4000,
                     max_seq_len=999, # train on sequences of 1000 (so 999 + teacher forcing shift)
                     input_seq_width=34,    # width of our data (32 + 3 actions)
                     output_seq_width=32,    # width of our data is 32
                     rnn_size=256,    # number of rnn cells
                     batch_size=10,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=5,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)

def build_hps(arglist):
  return HyperParams(num_steps=4000,
                     max_seq_len= arglist.batch_size -1, # train on sequences of 1000 (so 999 + teacher forcing shift)
                     input_seq_width=32 +arglist.action_space+ int(arglist.action_space*arglist.timestep*(arglist.agent_num - 1)),    # width of our data (32 + 3 actions)
                     output_seq_width=32,    # width of our data is 32
                     rnn_size=256,    # number of rnn cells
                     batch_size=10,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=5,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)

if __name__ == '__main__':
    arglist = parse_args()
    if not os.path.exists(arglist.model_save_path):
        os.makedirs(arglist.model_save_path)
    if not os.path.exists(arglist.initial_z_save_path):
        os.makedirs(arglist.initial_z_save_path)

    hps_model = build_hps(arglist)
    
    hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)

    raw_data = np.load(os.path.join(arglist.series_dir, "series.npz"))
    # load preprocessed data
    if arglist.use_vae:
      data_mu = raw_data["mu"]
      data_logvar = raw_data["logvar"]
    else:
      data_obs = raw_data["obs"]  
    data_action =  raw_data["action"]
    data_oppo_action = raw_data["oppo_action"]
    max_seq_len = hps_model.max_seq_len

    N_data = len(data_action) # should be 10k
    batch_size = hps_model.batch_size
    
    # save 1000 initial mu and logvars:
    if arglist.use_vae:
      initial_mu = np.copy(data_mu[:1000][ 0][:]*10000).astype(np.int).tolist()
      initial_logvar = np.copy(data_logvar[:1000][0][:]*10000).astype(np.int).tolist()
      with open(os.path.join("tf_initial_z", "initial_z.json"), 'wt') as outfile:
        json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))

    reset_graph()
    rnn = MDNRNN(hps_model)

    # train loop:w
    hps = hps_model
    start = time.time()
    for local_step in range(hps.num_steps):

      step = rnn.sess.run(rnn.global_step)
      curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate

      raw_z, raw_a ,raw_oppo_a= random_batch(arglist)
      print(raw_z.shape, raw_a.shape, raw_oppo_a.shape)
      raw_oppo_a = raw_oppo_a.reshape((len(raw_z), arglist.batch_size, -1 ))
      print(raw_z.shape, raw_a.shape, raw_oppo_a.shape)
      inputs = np.concatenate((raw_z[:, :-1, :], raw_a[:, :-1, :], raw_oppo_a[:, :-1, :]), axis=2)
      outputs = raw_z[:, 1:, :] # teacher forcing (shift by one predictions)

      feed = {rnn.input_x: inputs, rnn.output_x: outputs, rnn.lr: curr_learning_rate}
      (train_cost, state, train_step, _) = rnn.sess.run([rnn.cost, rnn.final_state, rnn.global_step, rnn.train_op], feed)
      if (step%20==0 and step > 0):
        end = time.time()
        time_taken = end-start
        start = time.time()
        output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, train_cost, time_taken)
        print(output_log)
        rnn.save_json(os.path.join(arglist.model_save_path, "rnn.json"))
    # save the model (don't bother with tf checkpoints json all the way ...)
    rnn.save_json(os.path.join(arglist.model_save_path, "rnn.json"))
