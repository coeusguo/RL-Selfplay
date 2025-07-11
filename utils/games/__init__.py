from .gomoku import GomokuGame

def get_game(args):
    if args.game == "gomoku":
        return GomokuGame(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    from argument import parse_args
    import numpy as np
    args = parse_args()
    game = get_game(args)

