
import math
import random
import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple
from collections import defaultdict
from .base import Agent

class MCTS:
    def __init__(self, args, game, policy):
        self.game = game
        self.policy = policy
        self.puct_factor = args.puct_factor
        self.num_tree_search = args.num_tree_search
        
        # frequency N(s) and N(s, a)
        self.N_s = defaultdict(int)
        self.N_sa = defaultdict(int)

        # Q(s, a)
        self.Q_sa = defaultdict(float)
        
        # pi(s, *):
        self.pi_s = defaultdict(None)

        # valide moves at state s
        self.valid_move_s = defaultdict(int)

        # store the terminal rewards if s is terminal state
        self.terminal_reward = defaultdict(None)

    def clear_search_tree(self):
        self.N_s.clear()
        self.N_sa.clear()
        self.Q_sa.clear()
        self.pi_s.clear()
        self.valid_move_s.clear()
        self.terminal_reward.clear()
        
    def tree_search(self, canonical_board: np.ndarray) -> np.ndarray:
        '''
        since our game board is very small (< 10 x 10)
        the computation of MCTS will be on CPU (except for executing policy model)
        '''

        # default player_id = 1 (-1 for opponent)
        cur_player_id = 1

        # convert 2D integer array to byte for dict hash operations
        s = canonical_board.tobytes()

        '''
        check if s is a terminal state
        None if not terminal
        1 for win, -1 for lose, 0 for draw
        '''
        self.terminal_reward[s] = self.game.is_terminal(canonical_board, cur_player_id)

        # return rewards if S is a terminal state
        if self.terminal_reward[s] is not None:
            return self.terminal_reward[s]

        if not s in self.pi_s: # not visited before, take as leaf node
            # get predicted policy (pi) and state value (v)
            pi, v = self.policy.predict(canonical_board)

            # check valid moves at state (s)
            valid_moves = self.game.get_valid_moves(canonical_board)

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
        
        # not a leaf node, compute PUCT
        pi_s = self.pi_s[s]
        highest_u = -float('inf')
        best_action = None
        for action_id, is_valid in enumerate(self.valid_move_s[s]):
            if not is_valid:
                continue
            
            # compute ucb
            ucb = self.Q_sa[(s, action_id)] + self.puct_factor * pi_s[action_id] * \
                    math.sqrt(self.N_s[s]) / (self.N_sa[(s, action_id)] + 1)
            # update highest Q
            if ucb > highest_u:
                highest_u = ucb
                best_action = action_id

        assert best_action is not None

        # get the maximized opponent state value
        s_opponent, opponent_id = self.game.get_next_state(canonical_board, cur_player_id, best_action)
        opponent_v = self.tree_search(s_opponent)

        # update Q(s, a)
        s_opponent = self.game.get_canonical_form(s_opponent, opponent_id)
        self.Q_sa[(s, best_action)] = (self.Q_sa[(s, best_action)] * self.N_s[s] - opponent_v) / \
                                      (self.N_s[s] + 1)

        # increment N(s) and N(s, a) by 1
        self.N_s[s] += 1
        self.N_sa[(s, best_action)] += 1

        # return the state value of child node of (s), by taking the best action
        return -opponent_v
    

    def action_probs(self, canonical_board: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        s = canonical_board.tobytes()

        # perform tree search
        for _ in range(self.num_tree_search):
            self.tree_search(canonical_board)

        valid_moves = self.valid_move_s[s]
        visting_freq = np.zeros_like(valid_moves)
        for action_id, is_valid in enumerate(valid_moves):
            if not is_valid:
                continue
            visting_freq[action_id] = self.N_sa[(s, action_id)]
        
        # sanity check
        assert self.N_s[s] == np.sum(visting_freq), f"{self.N_s[s]=}, {np.sum(visting_freq)=}"

        if temperature == 0:# deterministic
            # in case of multiple (s, a) pairs have same visiting frequency
            highest_freq = np.max(visting_freq)
            best_actions = np.array(np.argwhere(visting_freq == highest_freq)).reshape(-1)

            # randomly select 1 best actions (if there is multiple)
            best_action = np.random.choice(best_actions)
            probs = np.zeros_like(valid_moves)
            probs[best_action] = 1

        else:
            # smooth out the probs by temperature
            visting_freq = (visting_freq) ** (1 / temperature)
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

        # 

    def init_buffer(self):
        self.mcts.clear_search_tree()

    def run_one_episode(self) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
            if reward is not None:
                break

        # do some data augmentation
        canonical_boards = np.stack(raw_canonical_boards) # (B, N, N)
        canonical_boards = torch.from_numpy(canonical_boards).cuda()
        
        onehot_actions = F.one_hot(torch.tensor(action_ids),\
                        num_classes=self.game.get_action_size()).cuda() # (B, N*N)
        augmented_samples = self.game.get_symmetries(canonical_boards, onehot_actions)

        return self.wrap_up_samples(raw_samples, reward)

    def wrap_up_samples(self, raw_samples, reward, first_player = 1):

        '''
        raw_samples: [
            (boards_1, action_ids_1), ..., (boards_8, action_ids_8)
        ]

        post process samples,
        case:
            - reward = 1, user win, opponent lose
            - reward = -1, user lose, oppenent win
            - reward = 0, draw

            therefore:
            player_id = 1:
                reward = reward
            player_id = -1:
                reward = -reward

            reward = reward * player_id
        '''
        assert isinstance(raw_samples, list)

        def tensor_to_numpy(board, pi):
            if isinstance(board, torch.Tensor):
                board = board.cpu().numpy()
                pi = pi.cpu().numpy()
            
            # (B, n, n) to a list of B (n, n) np arrays
            return list(board), list(pi)

        num_steps = raw_samples[0].shape[0]

        player_ids = np.ones(num_steps, dtype=np.int64) * first_player
        player_ids[1::2] *= -1
        group_rewards = list(reward * player_ids)

        samples = []
        for sample_group in raw_samples:
            board, pi = tensor_to_numpy(*sample_group)
            

            samples.extend(list(zip(board, pi, group_rewards)))
            
        return samples