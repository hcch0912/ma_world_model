from env import PreyPredatorEnv 
import numpy as np 
import time
if __name__ == '__main__':
	env = PreyPredatorEnv()
	obs = env.reset() 
	while 1:
		env.render(mode = "human" ,image = obs)
		
		obs, reward, done, info = env.step([[[np.random.uniform(-1,1),np.random.uniform(-1,1)]],
			[[np.random.uniform(-1,1), np.random.uniform(-1,1) ],
			[np.random.uniform(-1,1), np.random.uniform(-1,1) ],
			[np.random.uniform(-1,1), np.random.uniform(-1,1) ],
			[np.random.uniform(-1,1), np.random.uniform(-1,1) ]]])
		print(reward)