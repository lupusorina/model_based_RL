import gymnasium as gym
import numpy as np
import go1_mujoco_env
import torch
from model import WorldModel
from cem import CEM
from system import System

collect_data = False

env = gym.make('gymnasium_env/GO1', xml_file='./xmls/scene_mjx_fullcollisions_flat_terrain.xml', render_mode = 'human', collect_data = collect_data)
observation, info = env.reset()
episode_over = False

nqpos = 7 + 12 
nqvel = 6 + 12 
naction = 12 
nreward = 1

# laod trained model output by train.py
#TODO: not hardcode this
PATH = '/home/zhonghezheng13579/research_projects/model_based_RL/robot_stand_up/models/model_20250427_140239_0'

loaded_model = WorldModel((nqpos + nqvel + naction), 128, (nqpos + nqvel + nreward))

loaded_model.load_state_dict(torch.load(PATH))

system = System(loaded_model, naction)

horizon = 5
num_particles = 100
num_iterations = 3
num_elite = 1000

cem = CEM(horizon, num_particles, num_iterations, num_elite, np.ones((horizon, naction)), np.ones((horizon, naction, naction)))
wait = True
while wait:
    best_act, best_actions, best_value = cem.optimize(observation, system, dt = 0.04, goal = None, damping_map=None, terrain_map=None, NN=None)
    observation, reward, terminated, truncated, info = env.step(best_act[0])

env.close()