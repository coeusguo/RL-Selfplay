
import math
import random
import numpy as np
from typing import Tuple
from collections import defaultdict

class MCTS:
    def __init__(self, args, game, policy):
        self.game = game
        self.policy = policy
        self.puct_factor = args.puct_factor
        self.num_tree_search = args.num_tree_search
        
        # frequency N(s) and N(s, a)
        self.N_s = defaultdict(0)
        self.N_sa = defaultdict(0)

        # Q(s, a)
        self.Q_sa = defaultdict(0)
        
        # pi(s, *):
        self.pi_s = defaultdict(None)

        # valide moves at state s
        self.valid_move_s = defaultdict(0)

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

        if self.pi_s[s] is None: # not visited before, take as leaf node
            # get predicted policy (pi) and state value (v)
            pi, v = self.policy(s)

            # check valid moves at state (s)
            valid_moves = self.game.get_valid_move(canonical_board, cur_player_id)

            # mask out invalid moves of pi
            pi = pi * valid_moves 

            # rescale the sum of valid move probs to 1
            rescale_factor = np.sum(pi)
            if rescale_factor > 1e-4: 
                # prob of valid moves are large enough
                self.pi_s = pi / rescale_factor
            else:
                # prob of valid moves are too small, make it uniform
                self.pi_s = valid_moves / np.sum(valid_moves)
            
            # store valid moves at state (s)
            self.valid_move_s[s] = valid_moves
            
            # take as leaf node, return state estimate
            return v

        # not a leaf node, compute PUCT
        pi_s = self.pi_s[s]
        highest_u = -100000
        best_action = None
        for action_id, is_valid in enumerate(self.valid_move_s[s]):
            if not is_valid:
                continue
            
            # compute ucb
            ucb = self.Q_sa[(s, action_id)] + self.puct_factor * pi_s[action_id] * \
                    math.sqrt(self.N_s(s)) / (self.N_sa + 1)
            
            # update highest Q
            if ucb > highest_u:
                highest_u = ucb
                best_action = action_id

        # get the maximized opponent state value
        s_opponent, opponent_id = self.game.get_next_state(canonical_board, cur_player_id, best_action)
        opponent_v = self.tree_search(s_opponent)

        # update Q(s, a)
        s_opponent = self.game.get_canonical_form(s_opponent, opponent_id)
        self.Q_sa[(s, best_action)] = (self.Q_sa[(s, best_action)] * self.N_s(s) - opponent_v) / \
                                      (self.N_sa[(s, best_action) + 1])

        # increment N(s) and N(s, a) by 1
        self.N_s[s] += 1
        self.N_s[(s, best_action)] += 1

        # return the state value of child node of (s), by taking the best action
        return -opponent_v
    

    def action_probs(self, canonical_board: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        s = canonical_board.tobytes()

        # perform tree search
        for _ in range(self.num_tree_search):
            self.search(canonical_board)

        valid_moves = self.valid_move_s[s]
        visting_freq = np.zeros_like(valid_moves)
        for action_id, is_valid in enumerate(valid_moves):
            if not is_valid:
                continue
            visting_freq = self.N_sa[(s, action_id)]
        
        # sanity check
        assert self.N_s[s] == np.sum(visting_freq)

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
            visting_freq = visting_freq ** (1 / temperature)
            probs = visting_freq / np.sum(visting_freq)

        return probs
    

class MCTSAgent:
    def __init__(self, args, policy, game):
        self.game = game
        self.policy = policy
        self.mcts = MCTS(args, game, policy)

        self.decay_step = args.temp_decay_step
        self.decay_rate = args.temp_decay_rate

    def get_policy_net(self):
        return self.policy

    def run_one_episode(self) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        
        step = 0
        raw_samples = []
        board = self.game.get_empty_board()
        self.mcts.clear_search_tree()

        player_id = 1
        init_player_id = player_id
        
        temp = self.temperature
        while True:
            step += 1
            canonical_board = self.game.get_canonical_form(board, player_id)

            # decay the temperature if needed
            if step >= self.temp_decay_step:
                temp = temp * self.temp_decay_rate

            pi = self.mcts.action_probs(canonical_board, temperature=temp)

            # do some data augmentation before adding to samples
            augmented_samples = self.game.get_symmetries(board, pi)
            samples.append((augmented_samples, player_id))

            # randomly sample an action from categorical distribution
            action_id = np.random.choice(np.arange(len(pi)), p=pi)

            # get updated board and opponent's id
            board, player_id = self.game.get_next_state(board, player_id, action_id)
            
            # check terminal condition
            reward = self.game.is_terminal(board, init_player_id)
            if reward is not None:
                break

        '''
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

        samples = []
        for sample_group in raw_samples:
            groups, player_id = sample_group
            board, pi = zip(*groups)
            group_rewards = np.array(reward * player_id)
            group_rewards = [r for _ in range(len(board))]

            samples.extend(list(zip(board, pi, group_rewards)))

        return samples