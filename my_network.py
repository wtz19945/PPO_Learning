import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(FeedForwardNN, self).__init__()

        # define layers
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    # Define forward evaluation of the network
    def forward(self, obs):
        # Convert observation to tensor in case it is numpy array
        if isinstance(obs, np.ndarray):
            # obs = torch.tensor(obs, dtype = torch.float)
            obs_array = np.array(obs)
            obs = torch.tensor(obs_array, dtype=torch.float)
        
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        output = self.layer3(x)

        return output
    
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    obs = torch.rand(1, 20)
    obs = obs.to(device)
    model = FeedForwardNN(20, 10).to(device)
    out = model(obs)
    print(out)



