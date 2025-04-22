import gymnasium as gym
import numpy as np
import go1_mujoco_env
import csv


collect_data = True

env = gym.make('gymnasium_env/GO1', xml_file='./xmls/scene_mjx_fullcollisions_flat_terrain.xml', collect_data = collect_data)

observation, info = env.reset()
episode_over = False

num_iterations = 500
T = 50
with open('control_data.csv', 'w', newline='') as file:
    fieldnames = ['IterNum', '(CurrState, Action)', '(NextState, Reward)']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader() # write the header row

    for i in range(num_iterations):

        for t in range(50):
            action = np.array([-0.352275, 1.18554, -2.80738, 
                            0.360892, 1.1806, -2.80281, 
                            -0.381197, 1.16812, -2.79123, 
                            0.391054, 1.1622, -2.78576])
    
            observation, reward, terminated, truncated, info = env.step(action)
            
        for t in range(T):
            action = np.array([-0.352275, 1.18554, -2.80738, 
                            0.360892, 1.1806, -2.80281, 
                            -0.381197, 1.16812, -2.79123, 
                            0.391054, 1.1622, -2.78576]) * (1- t/T) + \
                    np.array([0, 0.82, -1.63, 
                            0, 0.82, -1.63, 
                            0, 0.82, -1.63, 
                            0, 0.82, -1.63]) * t/T
            observation, reward, terminated, truncated, info  = env.step(action)

            if collect_data:     
                writer.writerow({'IterNum':i, '(CurrState, Action)': info['(CurrState, Action)'], '(NextState, Reward)': info['(NextState, Reward)']})

        # episode_over = terminated or truncated

env.close()