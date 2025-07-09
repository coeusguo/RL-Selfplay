from .loss import naive_loss

def get_loss(args):
    if args.loss_fn == "imatation":
        return naive_loss
    else:
        raise NotImplementedError