import torch 
import torch.nn as nn
import numpy as np 

class DQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1056, 512),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, x):
        try:
            temp = x.shape[2]
            x = x.view(x.shape[0], -1).float()
        except:
            x = x.view(1, -1).float()
        return self.fc(x)
