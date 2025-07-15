import torch

def cross_entropy_loss(log_pi, target_pi):
    # size: (B, A)
    assert log_pi.size() == target_pi.size()
    loss = -torch.sum(target_pi * log_pi, dim=-1, keepdim=False)
    return torch.mean(loss)

def mse_loss(predict, target):
    # size (B, 1)
    assert predict.size() == target.size()
    loss = (predict - target) ** 2
    return torch.mean(loss)

def imatiation_loss(pi, v, target_pi, target_v):
    policy_loss = cross_entropy_loss(pi, target_pi)
    value_loss = mse_loss(v, target_v)
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "loss": value_loss + policy_loss
    }



def ppo_pi_loss(log_pi, old_pi, advantages, clip_range):
    ratio = torch.exp(log_pi - torch.log(old_pi))

    advantages = advantages.view(-1, 1) # (B, 1)
    positive_adv_mask = advantages > 0

    upper_bound = torch.full_like(ratio, 1 + clip_range)
    lower_bound = torch.full_like(ratio, 1 - clip_range)
    cliped_ratio = torch.where(positive_adv_mask, 
                        torch.min(ratio, upper_bound), torch.max(ratio, lower_bound))

    loss = -cliped_ratio * advantages
    
    # compute clip statistics for logging
    pos_cliped_mask = ((ratio > 1 + clip_range) & positive_adv_mask) 
    neg_clipped_mask = ((ratio < 1 - clip_range) & ~positive_adv_mask)

    pos_rate = positive_adv_mask.sum() / positive_adv_mask.numel()
    neg_rate = (~positive_adv_mask).sum() / positive_adv_mask.numel()

    token_pos_cls = ratio.numel() * pos_rate
    token_neg_cls = ratio.numel() * neg_rate

    return {
        "ppo_pi_loss": loss,
        "pos_clip_ratio": pos_cliped_mask.sum() / (token_pos_cls+ 1e-5),
        "neg_clip_ratio": neg_clipped_mask.sum() / (token_neg_cls + 1e-5) 
    }

def ppo_value_loss(value, reward):
    advantages = reward - value
    return {
        "advantages": advantages,
        "value_loss": torch.mean(advantages ** 2)
    }
    

def ppo_loss(args, log_pi, old_pi, value, reward):
    # compute ppo value loss
    ppo_values = ppo_value_loss(value, reward)

    # compute ppo pi loss
    advantages = ppo_values.pop("advantages")
    clip_range = args.clip_range
    ppo_pi = ppo_pi_loss(log_pi, old_pi, advantages, clip_range)

    # update loss and print attributes
    pi_loss = ppo_pi.pop("ppo_pi_loss")
    v_loss = ppo_values["value_loss"]
    loss = {"loss": pi_loss + v_loss}

    loss.update(ppo_values)
    loss.update(ppo_pi)
    
    return loss