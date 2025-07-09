import torch

import torch
import torch.nn as nn
import torch.optim as optim

def get_optimizer(args, model: nn.Module):
    if args.optimizer.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError
    