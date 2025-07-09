import torch

def cross_entropy_loss(predict, target):
    # size: (B, A)
    assert predict.size() == target.size()
    loss = -torch.sum(target * predict, dim=-1, keepdim=False)
    return torch.mean(loss)

def mse_loss(predict, target):
    # size (B, 1)
    assert predict.size() == target.size()
    loss = (predict - target) ** 2
    return torch.mean(loss)

def imatiation_loss(pi, v, target_pi, target_v):
    policy_loss = cross_entropy_loss(pi, target_pi)
    value_loss = mse_loss(v, target_v)
    return policy_loss + value_loss

def ppo_loss(pi, old_pi):
    pass