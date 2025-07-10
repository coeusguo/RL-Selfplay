from .naive_policy import NaivePolicyNet
from .wrapper import Wrapper

def get_wrapped_policy(args, action_size: None | int = None):
    if action_size is None:
        print("no action_size provided, use board_size x board_size as action size")
        action_size = args.board_size * args.board_size
        
    if args.model == "naive":
        policy = NaivePolicyNet(args, action_size)
        return Wrapper(policy)
    else:
        raise NotImplementedError