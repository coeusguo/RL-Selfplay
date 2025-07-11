import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

NUM_CHANNELS = 512

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out + x

class NaivePolicyNet(nn.Module):
    def __init__(self, args, action_size):
        super(NaivePolicyNet, self).__init__()

        self.board_size = args.board_size
        self.action_size = action_size
        self.dropout = args.dropout
        
        self.conv1 = nn.Conv2d(1, NUM_CHANNELS, 3, stride=1, padding=1)
 
        self.block = nn.Sequential(
            ResidualBlock(NUM_CHANNELS, NUM_CHANNELS),
            ResidualBlock(NUM_CHANNELS, NUM_CHANNELS)
        )

        self.pi_fc = nn.Linear(NUM_CHANNELS, self.action_size)
        self.v_fc = nn.Linear(NUM_CHANNELS, 1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, s):
        B = s.size(0)
        out = self.conv1(s)
        out = self.block(out)

        out = self.gap(out).reshape(B, NUM_CHANNELS)

        pi_logits = self.pi_fc(out)
        v_logits = self.v_fc(out).view(B) # reshape (B, 1) to (B, )

        return F.log_softmax(pi_logits, dim=-1), torch.tanh(v_logits)
    
