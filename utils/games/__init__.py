from .gomoku import GomokuGame

def get_game(args):
    if args.game == "gomoku":
        return GomokuGame(args)
    else:
        raise NotImplementedError