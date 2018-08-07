# for the purpose of creating visualizations

import numpy as np
import gym

from scipy.misc import imresize as resize
from scipy.misc import toimage as toimage
from gym.spaces.box import Box
from two_player.pong import PongGame

SCREEN_Y = 64
SCREEN_X = 64
FACTOR =1

def _process_frame(frame):
  obs = frame[0:84, :, :].astype(np.float)/255.0
  obs = resize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs

class PongWrapper(PongGame):
  def __init__(self):
    super(PongWrapper, self).__init__(competitive = False)
    self.observation_space = Box(low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3))
    self.custom_viewer = None
    self.frame_count = 0
    self.current_frame = None
    self.vae_frame = None

  def step(self, action):
    obs, reward,[act1,act2], goals, win = super(PongWrapper, self).step(action)
    self.current_frame = _process_frame(obs)
    return self.current_frame, reward, win, {}

  def render(self, mode='human', close=False):

    if mode == "state_pixels":
      return super(PongWrapper, self).render("state_pixels")

    img_orig = self.current_frame

    img_vae = self.vae_frame

    img = np.concatenate((img_orig, img_vae), axis=1)

    img = resize(img, (int(np.round(SCREEN_Y*FACTOR)), int(np.round(SCREEN_X*FACTOR))*2))

    #img = img_orig

    if self.frame_count > 0:
      pass
      #toimage(img, cmin=0, cmax=255).save('output/'+str(self.frame_count)+'.png')
    self.frame_count += 1

    return super(PongWrapper, self).render(mode=mode, image = img, close=close)



class PPWrapper(PreyPredatorEnv):
  def __init__(self):
    super(PPWrapper, self).__init__()
    self.observation_space = Box(low = 0, high = 255, shape=(SCREEN_X, SCREEN_Y, 3))
    self.custom_viewer = None
    self.frame_count = 0
    self.current_frame = None
    self.vae_frame = None
  def step(self, action):
    obs, reward, done, _ = super(PPWrapper, self).step(action)
    self.current_frame = _process_frame(obs)
    return self.current_frame, reward, win, {}
  def render(self, mode = 'human', close = False):
    if mode == "state_pixels":
      return super(PPWrapper, self).render_mode("state_pixels")
    img_orig = self.current_frame

    img_vae = self.vae_frame

    img = np.concatenate((img_orig, img_vae), axis=1)

    img = resize(img, (int(np.round(SCREEN_Y*FACTOR)), int(np.round(SCREEN_X*FACTOR))*2))
    if self.frame_count > 0:
      pass
      #toimage(img, cmin=0, cmax=255).save('output/'+str(self.frame_count)+'.png')
    self.frame_count += 1
    return super(PPWrapper, self).render(mode = mode, image = img, close= close)
  def make_env(env)  :
    env = PreyPredatorEnv()
    if (seed >=0):
      env.seed(seed)
    return env  

def make_env(env_name, seed=-1, render_mode=False):
  if env_name == "Pong-2p-v0":
    env = PongWrapper()
  if env_name == "prey_predator":
    env = PPWrapper()  
  if (seed >= 0):
    env.seed(seed)
  '''
  print("environment details")
  print("env.action_space", env.action_space)
  print("high, low", env.action_space.high, env.action_space.low)
  print("environment details")
  print("env.observation_space", env.observation_space)
  print("high, low", env.observation_space.high, env.observation_space.low)
  assert False
  '''
  return env

    
