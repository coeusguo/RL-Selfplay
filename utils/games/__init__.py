from .gomoku import GomokuGame
from .gui import GUI

def get_game(args):
    if args.game == "gomoku":
        return GomokuGame(args)
    else:
        raise NotImplementedError

def get_gui(args, bot):
    return GUI(args, bot)

if __name__ == "__main__":
    from argument import parse_args
    import numpy as np
    args = parse_args()
    game = get_game(args)

