import os
import argparse
import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    # game setup
    parser.add_argument("-game", type=str, default="gomoku")
    parser.add_argument("-board-size", type=int, default=8)
    parser.add_argument("-num-in-a-row", type=int, default=4, 
                        help="the number of pieces in a row to win the game")

    # training related
    parser.add_argument("-main-epoch", type=int, default=100)
    parser.add_argument("-num-mini-epoch", type=int, default=10, 
                        help="for each epoch, the samples will trained for num-mini-epochs")
    parser.add_argument("-loss-fn", type=str, default="imatation")
    parser.add_argument("-lr", type=float, default=5e-4)
    parser.add_argument("-weight-decay", type=float, default=0.1)
    parser.add_argument("-dropout", type=float, default=0.0)
    parser.add_argument("-optimizer", type=str, default="adamw")
    parser.add_argument("-batch-size", type=int, default=64)
    parser.add_argument("-eval-every", type=int, default=2, help="how many epochs per evluate")
    parser.add_argument("-update-threshold", type=float, default=0.55,
                        help="when to update the rollout polcy? depends on the winning rate of new_policy"
                        "against the old polcy (default: 0.6, i.e. 55% winning rate)")
    parser.add_argument("-trainer-data-buffer-size", type=int, default=100, 
                    help="the total number of rollout samples buffered to train the policy model")

    # ray related
    parser.add_argument("-rollout-sample-buffer-size", type=int, default=5, 
                    help="maximum number of rollout refs to store in the buffer")

    # self play related
    parser.add_argument("-temperature", type=float, default=1.0)
    parser.add_argument("-temp-decay-step", type=int, default=20, 
                        help="from which step the temperature start to decay in each episode")
    parser.add_argument("-temp-decay-rate", type=float, default=0.1, 
                        help="temperature decay by a factor of given rate (0.0 for deterministic sampling)")
    parser.add_argument("-rollout-buffer-size", type=int, default=5, 
                        help="during aync. training, the maximum buffer size to store the rollout samples")
    parser.add_argument("-num-eval-matches", type=int, default=12, 
                        help="number of matches during evaluation")

    # mcts related
    parser.add_argument("-puct-factor", type=float, default=1.0)
    parser.add_argument("-num-tree-search", type=int, default=50, 
                         help="the number of tree searches per prediction")

    # policy model related
    parser.add_argument("-model", type=str, default="naive")
    parser.add_argument("-agent", type=str, default="mcts")

    # logging
    parser.add_argument("-log-dir", type=str, default="log")
    parser.add_argument("-save-every", type=int, default=10, help="how many epochs per evluate")

    # checkpointing
    parser.add_argument("-resume", action="store_true",  
                    help="resume training")
    parser.add_argument("-ckpt", type=str, default=None)

    args = parser.parse_args()

    # make unique log dir by time
    args.log_dir = Path(args.log_dir) / \
        f"{args.agent}_{args.game}_{args.board_size}x{args.board_size}_{datetime.datetime.now()}".replace(" ", "-")
    os.makedirs(args.log_dir, exist_ok=True)

    return args