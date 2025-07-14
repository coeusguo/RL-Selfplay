import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

NUM_CHANNELS = [128, 512, 512]
HEAD_DIM = 64

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential()
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out + self.shortcut(x)


class AttentionPool(nn.Module):
    def __init__(self, in_channel, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, in_channel))

        self.attn = nn.MultiheadAttention(
                        embed_dim=in_channel,
                        num_heads=HEAD_DIM,
                        batch_first=True,
                        dropout=dropout
                    )
        
        self.norm = nn.LayerNorm(in_channel)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.reshape(B, C, H * W).transpose(1, 2) # (B, HW, C)
        query = self.query.expand(B, 1, C) 

        q = torch.cat([x, query], dim=1)
        out, _ = self.attn(query=q, key=x, value=x)

        return self.norm(out[:, -1, :])


class NaivePolicyNet(nn.Module):
    def __init__(self, args, action_size, input_channel = 1):
        super(NaivePolicyNet, self).__init__()

        self.board_size = args.board_size
        self.action_size = action_size
        self.dropout = args.dropout
        
        self.conv1 = nn.Conv2d(input_channel, NUM_CHANNELS[0], kernel_size=3, stride=1, padding=1)
 
        self.block = nn.Sequential(*[
            ResidualBlock(NUM_CHANNELS[i], NUM_CHANNELS[i + 1])
            for i in range(len(NUM_CHANNELS) - 1)
        ])

        self.fc = nn.Sequential(
            nn.Linear(NUM_CHANNELS[-1], NUM_CHANNELS[-1]),
            nn.LayerNorm(NUM_CHANNELS[-1]),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )

        self.pi_fc = nn.Linear(NUM_CHANNELS[-1], self.action_size)
        self.v_fc = nn.Linear(NUM_CHANNELS[-1], 1)

        self.gap = AttentionPool(NUM_CHANNELS[-1])

    def forward(self, s):
        B = s.size(0)
        out = self.conv1(s)
        out = self.block(out)

        out = self.gap(out)
        out = self.fc(out)

        pi_logits = self.pi_fc(out)
        v_logits = self.v_fc(out).view(B) # reshape (B, 1) to (B, )

        return F.log_softmax(pi_logits, dim=-1), torch.tanh(v_logits)
    
