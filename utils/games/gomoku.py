import numpy as np
from typing import Literal, Tuple

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


class GomokuGame:
    idx_to_piece = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

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
        return (self.n, self.n)

    def get_action_size(self):
        # n * n possible stone places for gomoku
        return self.n * self.n

    def get_next_state(self, board: np.ndarray, 
                       player_id: int, action_id: int) -> Tuple[np.ndarray, int]:

        b = Board(self.n, board)

        # 1-D action to 2-D action
        move = (int(action_id / self.n), action_id % self.n)
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

    def is_terminal(self, board, player):
        '''
        check if any of the players wins,
        return None if not ended, 1 if player won, -1 if player lost, 0 if draw.
        '''
        
        def is_win(pieces: np.ndarray):
            if np.all(pieces == player):
                return 1
            if np.all(pieces == -player):
                return -1
            
            return 0
        
        reward = 0 # -1 for lose, 1 for win
        for idx in range(self.n * self.n):
            row = idx // self.n
            col = idx % self.n

            # empty
            if board[row, col] == 0: 
                continue

            # Check horizontal win
            if col + self.num_in_a_row <= self.n:
                horizontal_pieces = board[row, col: col + self.num_in_a_row]
                reward = is_win(horizontal_pieces)

            # Check vertical win
            if row + self.num_in_a_row <= self.n:
                vertical_pieces = board[row: row + self.num_in_a_row, col]
                reward = is_win(vertical_pieces)

            # Check diagonal (top-left to bottom-right) win
            if row + self.num_in_a_row <= self.n and col + self.num_in_a_row <= self.n:
                right_diag_pieces = board[np.arange(row, row + self.num_in_a_row), np.arange(col, col + self.num_in_a_row)]
                reward = is_win(right_diag_pieces)

            # Check diagonal (top-right to bottom-left) win
            if row + self.num_in_a_row <= self.n and col >= self.num_in_a_row:
                left_diag_pieces = board[np.arange(row, row + self.num_in_a_row), np.arange(col - self.num_in_a_row, col)[::-1]]
                reward = is_win(left_diag_pieces)

            if not reward == 0:
                break

        if reward == 0:
            # check if the board is full
            b = Board(self.n, board)
            if b.has_valid_moves():
                # the game is not end
                return None

        return reward

    def get_canonical_form(self, 
            board: np.ndarray, 
            player: Literal[1, -1]
        ) -> np.ndarray:
        '''
        may need to revert 1 and -1 as oppenent's view (player = -1)
        '''
        return player * board

    def get_symmetries(self, 
            board: np.ndarray, 
            pi: np.ndarray
        ) -> list[Tuple[np.ndarray, np.ndarray]]:
        '''
        this method is used for data augmentation
        number of possible rotation: {0, 90, 180, 270} degrees
        flip or not resulting in 2 possible states

        in total: 4 x 2 = 8 symmetrical board states and actions
        '''
        # convert 1-D pi into 2-D 
        pi = pi.reshape(self.n, self.n)

        augmented = []
        for num_degree in range(0, 4):
            rotated_board = np.rot90(board, num_degree)
            rotated_pi = np.rot90(pi, num_degree)
            augmented.append((rotated_board, rotated_pi.reshape(-1)))

            # also add flipped board and action
            flipped_board = np.fliplr(rotated_board)
            flipped_pi = np.fliplr(rotated_pi)
            augmented.append((flipped_board, flipped_pi.reshape(-1)))

        return augmented