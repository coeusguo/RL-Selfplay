# RL-Gomoku-Selfplay
training an agent to play chess  by RL with self-play

``sh script/train.sh`` to start the training, requires two GPUs

``sh script/pve.sh`` to run the user interface, requires one GPUs

With 2 RTX 2060 GPUs, it will take:

1. Around one week to train a 10 x 10 strong Gomoku (i.e. five-in-a-row) agent 
2. Less than 2 days to train a 8 x 8 strong Gomoku (i.e. five-in-a-row) agent 
3. Less than 1 minute to train a unbeatable tic-tac-toe agent

Training with pure MCTS takes ridiculusly long time when board size becomes larger.

Considering warmup with PPO first, then move to MCTS.


## done
1. Gomoku Chess board
2. MCTS Agent
3. RL Trainer, Rollout Sampler with Ray
4. Async. Training
5. Battle Arena
6. User interface

## todos
2. PPO baseline
3. PPO -> MCTS curicullum training
4. Other Chess games like Go and Othello
