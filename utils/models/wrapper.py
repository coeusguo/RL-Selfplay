import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class Wrapper(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def get_policy(self):
        return self.policy

    def load_state_dict(self, *args, **kwargs):
        self.policy.load_state_dict(*args, **kwargs)

    def predict(self, board: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:

        board = torch.FloatTensor(board, dtype=torch.float32)
        board = board.view(1, self.board_x, self.board_y)
        device = next(self.policy.parameters()).device
        board = board.to(device)

        self.policy.eval()
        with torch.no_grad():
            pi, v = self.policy(board)

        probs = torch.exp(pi).cpu().numpy()
        v= v.cpu().numpy()

        return probs, v

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)
