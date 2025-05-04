import numpy as np 
import torch 
class System:
    def __init__ (self, model, udim):
        self.model = model 
        self.udim = udim
        self.next_state = None 
        self.reward_val = None

    def action_lims(self):
        return np.ones((self.udim, self.udim))
    
    def dynamics(self, state, action, dt, damping_map=None, NN = None):
        input = list(state)+list(action)
        input = torch.tensor(input).float()
        out = self.model.forward(input).tolist()
        self.next_state = out[:-1]
        self.reward_val = out[-1]
        return self.next_state

    def reward(self, state, goal, action):
        return self.reward_val
