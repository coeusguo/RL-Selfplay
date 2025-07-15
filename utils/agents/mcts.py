
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

    def action_probs(self, 
        canonical_board: np.ndarray, 
        temperature: float = 1.0, 
        do_search: bool = True 
    ) -> np.ndarray:
        '''
            if do_search = False, make sure the input board has beed searched before,
            ususally set to False when need to collect soft probs during evluation, for training
        '''
        
        s = canonical_board.tobytes()

        # perform tree search
        if do_search:
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
        super(MCTSAgent, self).__init__(args, policy, game)

        self.mcts = MCTS(args, game, policy)
        self.temperature = args.temperature

        self.decay_step = args.temp_decay_step
        self.decay_rate = args.temp_decay_rate

    def init_buffer(self):
        self.mcts.clear_search_tree()

    def action_probs(self,
        canonical_board: np.ndarray, 
        temperature: float = 1.0, 
        do_search: bool = True 
    ):
        return self.mcts.action_probs(canonical_board, temperature=temperature, do_search=do_search)


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