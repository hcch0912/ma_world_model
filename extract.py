
import numpy as np
import random
import os
import gym

from model import make_model
from PIL import Image
import argparse
from util.make_env import *
import util.tf_util as U 
import collections

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
    parser.add_argument("--batch_size", type = int, default = 64, help = "set the batch_size")
    parser.add_argument("--max_episode_len", type = int, default = 50, help = "max episode length")
    parser.add_argument("--warm_up_steps", type = int, default = 1000, help = "set the warm up steps")
    parser.add_argument("--lr", type = float, default = 0.0001, help = "learning rate")
    parser.add_argument("--gamma", type = float, default = 0.99, help = "discount rate")
    parser.add_argument("--kl_tolerance", type = float, default = 0.5, help = "dk divergence tolerance")
    parser.add_argument("--data_dir", type = str,default = "./record")
    parser.add_argument("--model_save_path", type = str,default = "./tf_rnn/", help= "model save path")
    parser.add_argument("--z_size", type = int, default = 32, help = "z size")
    parser.add_argument("--initial_z_save_path", type = str, default = "tf_initial_z", help = "intial_z")
    parser.add_argument("--render_mode",type = bool, default = False, help = "render or not")
    parser.add_argument("--max_frames", type = int, default = 1000, help = "max frames to store")
    parser.add_argument("--max_trials", type = int, default = 200, help = "use this to extract one trial")
    parser.add_argument("--model_path", type = str, default = "", help = "path to load the model")
    parser.add_argument("--use_model", type = bool, default = False, help = "use model")
    parser.add_argument("--competitive", type = bool, default = False, help  = "competitive or cooperative")
    parser.add_argument("--screen_size", type = int, default = 200, help = "game screen size")
    parser.add_argument("--view_size", type = int, default = 100, help = "POMDP, partial obs size")
    parser.add_argument("--agent_num", type = int, default = 5, help = "total number of agent")
    parser.add_argument("--action_space", type = int, default = 2, help = "action space for each agent")
    parser.add_argument("--inference", type = bool, default = False, help = "use inference in policy or not")
    return parser.parse_args()



if __name__ == '__main__':


    arglist = parse_args()
    if not os.path.exists(arglist.data_dir):
        os.makedirs(arglist.data_dir)
    total_frames = 0
    if arglist.game == "Pong-2p-v0":
      with U.single_threaded_session():
        model = make_model(arglist, action_space= 1, scope = "pong", model_path =arglist.model_path,load_model = arglist.use_model)  
        env = make_env(arglist, full_episode=True)
        model.render_mode=arglist.render_mode
        U.initialize()
        for trial in range(arglist.max_trials): # 200 trials per worker
          try:
            random_generated_int = random.randint(0, 2**31-1)
            filename = arglist.data_dir+"/"+str(random_generated_int)+".npz"          
            recording_obs = []
            recording_action = []
            recording_oppo_action = []


            np.random.seed(random_generated_int)
            env.seed(random_generated_int)
            # random policy
            model.init_random_model_params(stdev=np.random.rand()*0.01)
            model.reset()
            obs = env.reset() # pixels

            for frame in range(arglist.max_frames):
              act_traj = collections.deque(np.zeros((arglist.timestep, arglist.action_space)), maxlen = arglist.timestep)
              if arglist.render_mode:
                env.render("human")
              else:
                env.render("rgb_array")
              obs = Image.fromarray(obs)
              obs = obs.resize((64,64),Image.ANTIALIAS)
              recording_obs.append(np.array(obs))

              z0, mu0, logvar0 = model.encode_obs(obs)
              # action shape (1,)
              action0 = model.get_action(z0)
              oppo_obs = np.flip(np.flip(obs,1),0)
              z1, mu1, logvar1 = model.encode_obs(oppo_obs)
              action1 = model.get_action(z1)

              act_traj.append(action1)
              recording_action.append(action0)
              recording_oppo_action.append(act_traj)
              obs, rewards, [act1, act2], goals, win = env.step([action0, action1])
              if win:
                break
            total_frames += (frame+1)
            print("dead at", frame+1, "total recorded frames for this worker", total_frames)
            recording_obs = np.array(recording_obs, dtype=np.uint8)
            recording_action = np.array(recording_action, dtype=np.float16)
            recording_oppo_action = np.array(recording_oppo_action, dtype = np.float16)
            np.savez_compressed(filename, obs=recording_obs, action=recording_action, oppo_actions = recording_oppo_action)
          except gym.error.Error:
            print("stupid gym error, life goes on")
            env.close()
            make_env(arglist,  render_mode=arglist.render_mode)
            continue      
        env.close()

    if arglist.game == "prey_predator":
      with U.single_threaded_session():     
          prey_model = make_model(arglist, action_space = 2, scope = "prey",model_path =arglist.model_path ,load_model = arglist.use_model)  
          predator_model = make_model(arglist, action_space = 2, scope = "predator", model_path = arglist.model_path, load_model = arglist.use_model)
          env = make_env(arglist)
          U.initialize()
          for trial in range(arglist.max_trials): # 200 trials per worker
            try:
              random_generated_int = random.randint(0, 2**31-1)
              prey_filename = os.path.join(arglist.data_dir, "prey",str(random_generated_int)+".npz")
              predator_filename = os.path.join(arglist.data_dir, "predator", str(random_generated_int)+".npz") 
              recording_obs = [[]] * 5
              recording_action = [[]] * 5
              recording_oppo_action = [[]]*5

              np.random.seed(random_generated_int)
              env.seed(random_generated_int)
              prey_model.init_random_model_params(stdev=np.random.rand()*0.01)
              predator_model.init_random_model_params(stdev=np.random.rand()*0.01)
              prey_model.reset()
              predator_model.reset()
              obs = env.reset() #
              for frame in range(arglist.max_frames):
                act_traj = [collections.deque(np.zeros((arglist.timestep, arglist.action_space)), maxlen = arglist.timestep)] *( arglist.agent_num)
                if arglist.render_mode:
                  env.render("human")
                else:
                  env.render("rgb_array")
                action_episode = []
                obs = [Image.fromarray(o) for o in obs]
                obs = [o.resize((64,64),Image.ANTIALIAS) for o in obs]
                obs = np.array([np.array(o) for o in obs])  
                for i in range(5):
                  recording_obs[i].append(obs[i])
                z0, mu0,  logvar0 = prey_model.encode_obs(obs[0])
                action0 = prey_model.get_action(z0)
                action_episode.append(action0)
                recording_action[0].append(action0)
                act_traj[0].append(action0)

                for i in range(1,5):
                  z1, mu1, logvar1 = predator_model.encode_obs(obs[i])
                  action1 = predator_model.get_action(z1)
                  action_episode.append(action1)
                  recording_action[i].append(action1) 
                  act_traj[i].append(action1)

                # print(np.array(act_traj).shape)  
                for  i in range(5):
                  recording_oppo_action[i].append(act_traj[i])
                obs, rewards, done, _ = env.step(action_episode)  
                if done: break
              total_frames += (frame+1)  
              print("dead at", frame+1, "total recorded frames for this worker", total_frames)
              recording_obs = np.array(recording_obs, dtype=np.uint8)
              recording_action = np.array(recording_action, dtype=np.float16)
              np.savez_compressed(prey_filename, obs=recording_obs[0], action=recording_action[0], oppo_actions = recording_oppo_action[1:])
              for i in range(1,5):
                oppo_actions = []
                oppo_actions.append(recording_oppo_action[0])
                for j in range(1,5):
                  if i ==j:
                    continue
                  else:
                    oppo_actions.append(recording_oppo_action[i])
                assert(len(oppo_actions)) == 4
                np.savez_compressed(predator_filename, obs=recording_obs[i], action=recording_action[i], oppo_actions = oppo_actions )
            except gym.error.Error:
              print("stupid gym error, life goes on")
              env.close()
              make_env(arglist, render_mode=arglist.render_mode)
              continue      
          env.close()     



  