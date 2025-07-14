python pve.py \
    -agent aync_mcts_opponent \
    -ckpt log/policy_best.pth \
    -num-tree-search 10 \
    -board-size 3 \
    -num-in-a-row 3 