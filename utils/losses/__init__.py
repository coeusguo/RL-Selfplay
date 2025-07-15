from .loss import imatiation_loss, ppo_loss

def get_loss(args):
    if args.loss_fn == "imatation":
        return imatiation_loss
    elif args.loss_fn == "ppo":
        return ppo_loss
    else:
        raise NotImplementedError