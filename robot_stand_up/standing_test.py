import gymnasium as gym
import numpy as np
from go1_mujoco_env import Go1MujocoEnv


env = gym.make('Ant-v5', xml_file='./xmls/scene_mjx_fullcollisions_flat_terrain.xml', ctrl_cost_weight=0.001, healthy_z_range=(0.48, 0.68), render_mode='human')

observation, info = env.reset()
episode_over = False

while not episode_over:

    T = 100
    for t in range(50):
        print("tucking ", t)
        action = np.array([-0.352275, 1.18554, -2.80738, 
                           0.360892, 1.1806, -2.80281, 
                           -0.381197, 1.16812, -2.79123, 
                           0.391054, 1.1622, -2.78576])
   
        observation, reward, terminated, truncated, info = env.step(action)
        
    for t in range(T):
        print("standing",  t)
        action = np.array([-0.352275, 1.18554, -2.80738, 
                           0.360892, 1.1806, -2.80281, 
                           -0.381197, 1.16812, -2.79123, 
                           0.391054, 1.1622, -2.78576]) * (1- t/T) + \
                 np.array([0, 0.82, -1.63, 
                           0, 0.82, -1.63, 
                           0, 0.82, -1.63, 
                           0, 0.82, -1.63]) * t/T
   
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    

env.close()