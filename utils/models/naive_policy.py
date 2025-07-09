import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

NUM_CHANNELS = 512

class NaivePolicyNet(nn.Module):
    def __init__(self, args, action_size):
       
        self.board_size = args.board_size
        self.action_size = action_size
        self.dropout = args.dropout


        super(NaivePolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, NUM_CHANNELS, 3, stride=1, padding=1)
 
        self.fc3 = nn.Linear(NUM_CHANNELS, self.action_size)
        self.fc4 = nn.Linear(NUM_CHANNELS, 1)

    def forward(self, s):
        pass
    
