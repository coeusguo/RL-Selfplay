import numpy as np
import tkinter as tk
import random
import time
from typing import Literal, Tuple
from .base import BoardGame
from tkinter import messagebox

class Board():
    '''
    1 = white, -1 = black, 0 = empty
    '''

    def __init__(self, board_size: int, existing_board: None | np.ndarray = None):

        self.n = board_size
        # Create the empty board array.
        self.pieces = np.zeros((self.n, self.n), dtype=np.int64)

        # or initialized from existing board
        if existing_board is not None:
            assert existing_board.shape == (self.n, self.n), f"invalid board size: {existing_board.shape}"
            self.pieces = np.copy(existing_board)

    def get_valid_moves(self) -> np.ndarray:
        # return all the empty places
        return np.argwhere(self.pieces == 0)

    def has_valid_moves(self):
        return len(self.get_valid_moves()) > 0

    def execute_move(self, move: Tuple[int, int], color: Literal[-1, 0, 1]):
        row, col = move
        self.pieces[row, col] = color


class GomokuGame(BoardGame):

    @staticmethod
    def get_piece_from_idx(idx: Literal[-1, 0, 1]):
        return GomokuGame.idx_to_piece[idx]

    def __init__(self, args):
        '''
        num_in_a_row: how many pieces in a row to win, typically 5, might be 4 or even lower
        '''
        self.n = args.board_size
        self.num_in_a_row = args.num_in_a_row

    def get_empty_board(self):
        b = Board(self.n)
        return np.array(b.pieces)

    def get_board_size(self):
        # square board of size n by default
        return self.n

    def get_action_size(self):
        # n * n possible stone places for gomoku
        return self.n * self.n

    def get_next_state(self, board: np.ndarray, 
                       player_id: int, action_id: int) -> Tuple[np.ndarray, int]:

        b = Board(self.n, board)

        # 1-D action to 2-D action
        move = (action_id // self.n, action_id % self.n)
        b.execute_move(move, player_id)

        # return next_state and opponent's id
        return (b.pieces, -player_id)

    def get_valid_moves(self, board: np.ndarray) -> np.ndarray:
        
        valid_moves = np.zeros_like(board)

        # returns a list of 2-D valid moves
        b = Board(self.n, board)
        moves_2d = b.get_valid_moves()
        
        # set all valid moves to 1
        rows, cols = moves_2d[:, 0], moves_2d[:, 1]
        valid_moves[rows, cols] = 1

        # reshape to 1-D
        return valid_moves.reshape(-1)

    def wins_here(self, board, position_1d):
        row, col = position_1d // self.n, position_1d % self.n

        player_id = board[row, col]
        assert player_id != 0, "the position you provided is empty!"

        for (d_row, d_col) in self.directions:
            count = 1

            # forward check
            for idx in range(1, self.num_in_a_row): 
                next_row = row + d_row * idx
                next_col = col + d_col * idx
                if next_row >= self.n or next_col >= self.n or next_row < 0 or next_col < 0:
                    break
                elif board[next_row, next_col] == player_id:
                    count += 1
                else: 
                    break
            
            # backward check
            for idx in range(1, self.num_in_a_row):
                next_row = row - d_row * idx
                next_col = col - d_col * idx

                if next_row >= self.n or next_col >= self.n or next_row < 0 or next_col < 0:
                    break
                elif board[next_row, next_col] == player_id:
                    count += 1
                else:
                    break

            if count >= self.num_in_a_row:
                return True
        
        return False

    def is_terminal(self, board, player: Literal[-1, 1]):
        '''
            check if any of the players wins,
            return: 
                None, if not ended, 
                1, if player won, 
                -1, if player lost 
                0, if draw.

            causion:
            if you passed in a canonical board, make sure player = 1 
        '''

        has_epmty_place = False
        for row in range(self.n):
            for col in range(self.n):
                # empty place
                if board[row, col] == 0:
                    has_epmty_place = True
                    continue

                # check if any one wins at this position
                if self.wins_here(board, row * self.n + col):
                    return 1 if board[row, col] == player else -1

        # no one wins but still has empty places -> game not terminated yet
        if has_epmty_place:
            return None

        # no one wins and no empty space left -> draw
        return 0
