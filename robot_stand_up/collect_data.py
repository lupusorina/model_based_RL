import gymnasium as gym
import numpy as np
import go1_mujoco_env
import csv


env = gym.make('gymnasium_env/GO1', xml_file='./xmls/scene_mjx_fullcollisions_flat_terrain.xml', collect_data = True)

observation, info = env.reset()
episode_over = False

num_iterations = 500
horizon = 50
act_size = 12
mean = [np.zeros(act_size) for i in range(horizon)]
covar = [0 for i in range(horizon)]

with open('control_data.csv', 'w', newline='') as file:
    fieldnames = ['IterNum', '(CurrState, Action)', '(NextState, Reward)']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader() # write the header row

    for i in range(num_iterations):

        for t in range(horizon):
            action = np.array([-0.352275, 1.18554, -2.80738, 
                            0.360892, 1.1806, -2.80281, 
                            -0.381197, 1.16812, -2.79123, 
                            0.391054, 1.1622, -2.78576])
    
            observation, reward, terminated, truncated, info = env.step(action)
            
        for t in range(horizon):
            action = np.array([-0.352275, 1.18554, -2.80738, 
                            0.360892, 1.1806, -2.80281, 
                            -0.381197, 1.16812, -2.79123, 
                            0.391054, 1.1622, -2.78576]) * (1- t/horizon) + \
                    np.array([0, 0.82, -1.63, 
                            0, 0.82, -1.63, 
                            0, 0.82, -1.63, 
                            0, 0.82, -1.63]) * t/horizon
            observation, reward, terminated, truncated, info  = env.step(action)

            # writing to file
            writer.writerow({'IterNum':i, '(CurrState, Action)': info['(CurrState, Action)'], '(NextState, Reward)': info['(NextState, Reward)']})

            mean[t] += action 
            covar[t] += action * action

        # episode_over = terminated or truncated

for i in range(horizon):
    mean[i] = (mean[i] / num_iterations).tolist()
    covar[i] = (covar[i] / num_iterations).tolist()

# Logging mean and covariance of sampled trajectories
with open('mean_and_covar.csv', 'w', newline='') as file:
    fieldnames = ['Mean', 'Covar']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader() # write the header row
    writer.writerow({'Mean': mean, 'Covar': covar})

env.close()