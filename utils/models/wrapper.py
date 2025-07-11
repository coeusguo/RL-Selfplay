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

    def state_dict(self):
        return self.policy.state_dict()

    def predict(self, board: np.ndarray | torch.Tensor)-> Tuple[np.ndarray, np.ndarray]:
        
        n, _ = board.shape
        device = next(self.policy.parameters()).device
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board).float()

        # todo: may need to accept a batch of examples in the future
        board = board.view(1, 1, n, n) # (B, C, H, W)
        board = board.to(device)

        self.policy.eval()
        with torch.no_grad():
            pi, v = self.policy(board)

        pi, v = pi.view(-1), v.view(-1)
        probs = torch.exp(pi).cpu().numpy()
        v= v.cpu().numpy()
        
        return probs, v

    def forward(self, s, *args, **kwargs):
        if s.ndim == 3: # (B, H, W)
            B, H, W = s.size()
            s = s.view(B, 1, H, W)
        else:
            assert s.ndim == 4, f"expected 4 dimsions (B, 1, H, W), got {s.size()}"
        return self.policy(s, *args, **kwargs)
