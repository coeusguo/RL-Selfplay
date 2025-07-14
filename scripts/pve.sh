python pve.py \
    -pve \
    -agent aync_mcts_opponent \
    -ckpt log/policy_best.pth \
    -num-tree-search 10 \
    -board-size 5 \
    -num-in-a-row 4 