from two_player.pong import PongGame
from prey_predator.env import PreyPredatorEnv



def make_env(arglist,  seed=-1, render_mode=False, full_episode=False):
    if arglist.game == "Pong-2p-v0":
      return PongGame(competitive = arglist.competitive)
    if arglist.game == "prey_predator":
      return PreyPredatorEnv( view_size = arglist.view_size )  
