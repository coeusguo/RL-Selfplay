from argument import parse_args
from utils.games import get_gui, get_game
from utils.agents import get_agent
from utils.models import get_wrapped_policy
from pathlib import Path
import torch
import os

def pve(args):
    policy = get_wrapped_policy(args)

    assert os.path.exists(args.ckpt), f"no ckpt found under args.ckpt: {args.ckpt}"
    state_dict = torch.load(Path(args.ckpt))
    policy.load_state_dict(state_dict)

    if torch.cuda.is_available():
        policy.cuda()

    game = get_game(args)
    bot = get_agent(args, policy, game)
    gui = get_gui(args, bot)
    
if __name__ == "__main__":
    args = parse_args()
    pve(args)
