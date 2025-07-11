import abc
import torch
import numpy as np
from typing import Literal, Tuple
from functools import partial

class BoardGame(abc.ABC):
    
    @abc.abstractclassmethod
    def get_action_size():
        pass

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
