from .loss import imatiation_loss

def get_loss(args):
    if args.loss_fn == "imatation":
        return imatiation_loss
    else:
        raise NotImplementedError