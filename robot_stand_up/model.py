import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModel(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(WorldModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out