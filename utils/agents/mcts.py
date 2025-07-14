
import math
import threading
import random
import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple
from collections import defaultdict
from .base import Agent
import time

class MCTS:
    def __init__(self, args, game, policy):
        self.game = game
        self.policy = policy
        self.puct_factor = args.puct_factor
        self.num_tree_search = args.num_tree_search
        self.action_size = game.get_action_size()
        self.board_size = game.get_board_size()
        
        # frequency N(s) and N(s, a)
        self.N_s = defaultdict(int)
        self.N_sa = defaultdict(lambda: np.zeros(self.action_size, dtype=np.int64))

        # Q(s, a)
        self.Q_sa = defaultdict(lambda: np.zeros(self.action_size, dtype=np.float32))
        
        # pi(s, *):
        self.pi_s = defaultdict(lambda: None)

        # valide moves at state s
        self.valid_move_s = defaultdict()

        '''
            -1 means not visited, 0 means draw, 1 means win
            since seart function always takes canonical board as input
            we only need to check if player 1 wins,
        '''
        self.terminal = defaultdict(lambda: None)

        self.num = 0

    def clear_search_tree(self, num_visits_only = False):
        self.N_s.clear()
        self.N_sa.clear()
        self.Q_sa.clear()

        if not num_visits_only:
            self.pi_s.clear()
            self.valid_move_s.clear()
            self.terminal.clear()
        
    def tree_search(self, canonical_board: np.ndarray) -> np.ndarray:
        '''
            since our game board is very small (< 10 x 10)
            the computation of MCTS will be on CPU (except for executing policy model)
        '''

        # default player_id = 1, due to canonical board as the input
        cur_player_id = 1

        # convert 2D integer array to byte for dict hash operations
        s = canonical_board.tobytes()

        if self.terminal[s] is not None:
            return self.terminal[s]

        if self.pi_s[s] is None: # not visited before, take as leaf node

            # check valid moves at state (s)
            valid_moves = self.game.get_valid_moves(canonical_board)

            if np.sum(valid_moves) == 0:
                # no valid moves, must be a draw state
                self.terminal[s] = 0
                return 0
            
            # get predicted policy (pi) and state value (v)
            pi, v = self.policy.predict(canonical_board)

            # mask out invalid moves of pi
            pi = pi * valid_moves 

            # rescale the sum of valid move probs to 1
            rescale_factor = np.sum(pi)
            if rescale_factor > 1e-4: 
                # prob of valid moves are large enough
                self.pi_s[s] = pi / rescale_factor
            else:
                # prob of valid moves are too small, make it uniform
                self.pi_s[s] = valid_moves / (np.sum(valid_moves) + 1e-5)
            
            # store valid moves at state (s)
            self.valid_move_s[s] = valid_moves
            # take as leaf node, return state estimate

            return v

        # multi-arm bandit, compute and pick the action with the highest UCB
        ucb = self.Q_sa[s] + self.puct_factor * self.pi_s[s] * np.sqrt(self.N_s[s]) / (self.N_sa[s] + 1)
        valid_ucb = np.where(self.valid_move_s[s], ucb, -np.inf)
        highest_ucb = np.max(valid_ucb)
        best_actions = np.flatnonzero(valid_ucb == highest_ucb)
        best_action = np.random.choice(best_actions)

        # get the maximized opponent state value
        updated_board, opponent_id = self.game.get_next_state(canonical_board, cur_player_id, best_action)

        # oppenent's turn 
        canonical_board = self.game.get_canonical_form(updated_board, opponent_id)

        if self.game.wins_here(updated_board, best_action):
            # check terminal state, updated_board is also canonical to cur_player_id
            self.terminal[canonical_board.tobytes()] = -1

        opponent_v = self.tree_search(canonical_board)

        # update Q(s, a)
        self.Q_sa[s][best_action] = (self.Q_sa[s][best_action] * self.N_sa[s][best_action] - opponent_v) / \
                                      (self.N_sa[s][best_action] + 1)

        # increment N(s) and N(s, a) by 1
        self.N_s[s] += 1
        self.N_sa[s][best_action] += 1

        # return the state value of child node of (s), by taking the best action
        return -opponent_v

    def action_probs(self, canonical_board: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        
        s = canonical_board.tobytes()

        # perform tree search
        for i in range(self.num_tree_search):
            self.num = 0
            self.tree_search(canonical_board)

        if temperature == 0:# deterministic
            # in case of multiple (s, a) pairs have same visiting frequency
            highest_freq = np.max(self.N_sa[s])
            best_actions = np.argwhere(self.N_sa[s] == highest_freq).reshape(-1)

            # randomly select 1 best actions (if there is multiple)
            best_action = np.random.choice(best_actions)
            probs = np.zeros(self.action_size)
            probs[best_action] = 1

        else:
            # smooth out the probs by temperature
            visting_freq = (self.N_sa[s]) ** (1 / temperature)
            probs = visting_freq / np.sum(visting_freq)

        return probs
    

class MCTSAgent(Agent):
    def __init__(self, args, policy, game):
        self.game = game
        self.policy = policy
        self.mcts = MCTS(args, game, policy)
        self.temperature = args.temperature

        self.decay_step = args.temp_decay_step
        self.decay_rate = args.temp_decay_rate

    def init_buffer(self):
        self.mcts.clear_search_tree()

    def run_one_episode(self, non_draw = False) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        self.policy.eval()

        # sample buffers
        raw_canonical_boards = []
        action_ids = []

        player_id = 1
        init_player_id = player_id
        board = self.game.get_empty_board()

        temp = self.temperature
        decayed = False
        step = 0
        while True:
            step += 1
            canonical_board = self.game.get_canonical_form(board, player_id)

            # decay the temperature if needed
            if step >= self.decay_step and not decayed:
                temp = temp * self.decay_rate
                decayed = True

            pi = self.mcts.action_probs(canonical_board, temperature=temp)

            # randomly sample an action from categorical distribution
            action_id = np.random.choice(np.arange(len(pi)), p=pi).item()

            # save samples
            raw_canonical_boards.append(canonical_board)
            action_ids.append(action_id)

            # get updated board and opponent's id
            board, player_id = self.game.get_next_state(board, player_id, action_id)
            
            # check terminal condition
            reward = self.game.is_terminal(board, init_player_id)
            if non_draw and reward == 0:
                # draw, but requires non-draw game, restart the game
                board = self.game.get_empty_board()
                raw_canonical_boards = []
                action_ids = []
                player_id = init_player_id
                continue

            if reward is not None:
                # print(self.game.get_readable_board(board))
                # print(f"{reward=}")
                break

        return self.pack_samples([raw_canonical_boards], [action_ids], [reward], [init_player_id])[0]

    def pack_samples(self, 
            canonical_boards: list[list[np.ndarray]], 
            action_ids: list[list[int]], 
            rewards: list[int], 
            init_player_ids: list[int]
        ):

        '''
        accept a batch of episodes

        post process samples,
        case:
            - reward = 1, init_player win, opponent lose
            - reward = -1, init_player lose, oppenent win
            - reward = 0, draw

        therefore, if:
            init_player_id = 1:
                reward = reward
            init_player_id = -1:
                reward = -reward

            reward = reward * player_id

        return 
        [ # episodes
            [ # 8 directions * episode_lengths
                (board, pi, reward), ...
            ]
        ] 
        '''
        assert len(canonical_boards) == len(action_ids) == len(rewards) == len(init_player_ids)

        episode_lengths = []
        _boards, _action_ids, _player_ids, _rewards = [], [], [], []
        for idx in range(len(action_ids)):
            # note length for each episode
            elength = len(action_ids[idx])
            episode_lengths.append(elength)

            _boards.append(np.stack(canonical_boards[idx])) # [(b1, n, n), (b2, n, n), ...]
            _action_ids.append(np.array(action_ids[idx], dtype=np.int64)) # (B,)

            player_ids = np.ones(elength, dtype=np.int64) * init_player_ids[idx]
            player_ids[1::2] = - init_player_ids[idx]
            _player_ids.append(player_ids)

            _rewards.append(np.ones(elength, dtype=np.int64) * rewards[idx])
        
        # B = b1 + b2 + ... + b_n, where b_i is the length of i-th episode
        boards = np.concatenate(_boards, axis=0) #(B, N, N)
        action_ids = np.concatenate(_action_ids, axis=0) #(B,)
        player_ids = np.concatenate(_player_ids, axis=0) #(B,)
        rewards = np.concatenate(_rewards, axis=0) #(B,)

        boards = torch.from_numpy(boards).float()
        action_ids = torch.from_numpy(action_ids)
        if torch.cuda.is_available():
            boards = boards.cuda()
            action_ids = action_ids.cuda()

        # do some data augmentation
        onehot_actions = F.one_hot(action_ids, num_classes=self.game.get_action_size()) # (B, N*N)
        augmented_samples = self.game.get_symmetries(boards, onehot_actions) # (8, B, N*N)

        break_points = np.cumsum(episode_lengths)[:-1]
        group_rewards = rewards * player_ids
        group_rewards = list(np.split(group_rewards, break_points))
        group_rewards = [list(rewards) for rewards in group_rewards] 

        samples = [[] for _ in range(len(episode_lengths))]

        for (board, pi) in augmented_samples:
            board = board.cpu().numpy()
            pi = pi.cpu().numpy()

            board = list(np.split(board, break_points))
            pi = list(np.split(pi, break_points))

            for idx in range(len(board)):

                board_ls, pi_ls = list(board[idx]), list(pi[idx])
                samples[idx].extend(list(zip(board_ls, pi_ls, group_rewards[idx])))

        return samples


class AyncMCTSOpponent(MCTSAgent):
    def __init__(self, args, policy, game, player_id = -1):
        super(AyncMCTSOpponent, self).__init__(args, policy, game)

        self.player_id = player_id
        self.stop_event = threading.Event()
        self.worker = None

        init_board = self.game.get_empty_board()
        self.endless_tree_search(init_board)

    def _search_loop(self, canonical_board):
        # this function will keep calling self.mcts.tree_search(canonical_board)
        while self.stop_event.is_set():
            self.mcts.tree_search(canonical_board)
            time.sleep(0.05)

    def endless_tree_search(self, canonical_board):

        if self.worker is not None and self.worker.is_alive():
            self.stop_event.clear()
            self.worker.join()

        # restart a new thread
        self.stop_event.set()
        self.worker = threading.Thread(
            target=self._search_loop, args=(canonical_board,)
        )
        self.worker.start()

    def move(self, board):
        canonical_board = self.game.get_canonical_form(board, self.player_id)

        # stop endless_tree_seach
        self.stop_event.clear()
        self.worker.join()

        # perform tree search again
        for _ in range(self.mcts.num_tree_search):
            self.mcts.tree_search(canonical_board)

        action_id = np.argmax(self.mcts.action_probs(canonical_board, temperature=0))

        board, next_player_id = self.game.get_next_state(board, self.player_id, action_id)
        canonical_board = self.game.get_canonical_form(board, next_player_id)

        # restart the endless tree search on new board
        self.endless_tree_search(canonical_board)

        return action_id