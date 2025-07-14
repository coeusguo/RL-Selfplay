import abc
import torch
import numpy as np
from typing import Literal, Tuple
from functools import partial

class BoardGame(abc.ABC):
    idx_to_piece = {
        -1: "X",
        +0: "-",
        +1: "O"
    }
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    @abc.abstractclassmethod
    def get_action_size():
        pass

    def get_board_size():
        return self.n

    def get_canonical_form(self, 
            board: np.ndarray, 
            player: Literal[1, -1]
        ) -> np.ndarray:
        '''
        may need to revert 1 and -1 as oppenent's view (player = -1)
        '''
        return player * board

    def get_symmetries(self, 
            board: np.ndarray | torch.Tensor, 
            pi: np.ndarray
        ) -> list[Tuple[np.ndarray, np.ndarray]]:
        '''
        this method is used for data augmentation
        number of possible rotation: {0, 90, 180, 270} degrees
        flip or not resulting in 2 possible states

        in total: 4 x 2 = 8 symmetrical board states and actions

        return: a list of 8 augmented (boards, actions) tuples
        '''
        original_board_dim = board.ndim
        if original_board_dim == 2: 
            board = board.reshape(1, self.n, self.n)
        else:
            assert original_board_dim == 3 # a batch of boards

        batch_size = board.shape[0]
        def restore(board, pi):
            if original_board_dim == 2:
                board = board.squeeze(0)
                pi = pi.reshape(-1)
            else:
                pi = pi.reshape(batch_size, -1)

            return board, pi

        if isinstance(board, torch.Tensor):
            rot90_fn = partial(torch.rot90, dims=[1, 2])
            flip_fn = partial(torch.flip, dims=[2])
        else:
            rot90_fn = partial(np.rot90, axes=(1, 2))
            flip_fn = partial(np.flip, axis=2)

        # convert 1-D pi into 2-D 
        pi = pi.reshape(batch_size, self.n, self.n)

        augmented = []
        for num_degree in range(0, 4):
            rotated_board = rot90_fn(board, num_degree)
            rotated_pi = rot90_fn(pi, num_degree)
            augmented.append(restore(rotated_board, rotated_pi))

            # also add flipped board and action
            flipped_board = flip_fn(rotated_board)
            flipped_pi = flip_fn(rotated_pi)
            augmented.append(restore(flipped_board, flipped_pi))

        return augmented

    def get_readable_board(self, board):
        """
        Creates a visually improved, grid-based board using Unicode
        box-drawing characters for a cleaner look.
        """
        # Use a middle dot for empty spaces and define piece characters
        
        # Define the box-drawing characters for the grid
        horizontal = "───"
        vertical = "│"
        top_left = "┌"
        top_right = "┐"
        bottom_left = "└"
        bottom_right = "┘"
        t_down = "┬"
        t_up = "┴"
        t_left = "├"
        t_right = "┤"
        cross = "┼"
        
        n = board.shape[0]
        cache = []

        # --- Build the board string row by row ---

        # Create the column headers (e.g., "    1   2   3 ")
        col_headers = [f"{i+1:^3}" for i in range(n)]
        cache.append("   " + " ".join(col_headers))

        # Create the top border of the grid (e.g., "  ┌───┬───┬───┐")
        top_border = "  ┌" + (horizontal + t_down) * (n - 1) + horizontal + "┐"
        cache.append(top_border)

        # Create each row with pieces and separators
        for r in range(n):
            # Format the row with pieces (e.g., " 1 │ O │ X │ · │")
            row_str = f"{r+1: >2}│"
            for c in range(n):
                piece = self.idx_to_piece[board[r, c]]
                row_str += f" {piece} │"
            cache.append(row_str)

            # Add a separator line or the final bottom border
            if r < n - 1:
                # e.g., "  ├───┼───┼───┤"
                separator = "  ├" + (horizontal + cross) * (n - 1) + horizontal + "┤"
                cache.append(separator)
            else:
                # e.g., "  └───┴───┴───┘"
                bottom_border = "  └" + (horizontal + t_up) * (n - 1) + horizontal + "┘"
                cache.append(bottom_border)

        return "\n".join(cache)
