python pve.py \
    -pve \
    -agent aync_mcts_opponent \
    -ckpt pretrained/9x9_c5.pth \
    -num-tree-search 600 \
    -board-size 9 \
    -num-in-a-row 5 