# RL-Gomoku-Selfplay
training an agent to play chess  by RL with self-play

``sh script/ray_train_c5.sh`` to train a 9x9 gomoku agent, requires two GPUs

``sh script/pve_9x9_c5.sh`` to run the user interface, requires one GPU

With 2 RTX 2060 GPUs, it will take:

1. Around one week to train a 12 x 12 strong five-in-a-row agent 
2. Less than 2 days to train a 9 x 9 strong five-in-a-row agent 
3. Less than 2 hours to train a 7 x 7 optimal four-in-a-row agent
3. Less than 1 minute to train a unbeatable tic-tac-toe agent

Training with pure MCTS takes ridiculusly long time when board size becomes larger.

Considering warmup with PPO first, then move to MCTS.

pretrained ckpt comming soon....


## done
1. Gomoku Chess board
2. MCTS Agent
3. RL Trainer, Rollout Sampler with Ray
4. Async. Training
5. Battle Arena
6. User interface
7. PPO Baseline

## todos
3. PPO -> MCTS curicullum training
4. Other Chess games like Go and Othello
