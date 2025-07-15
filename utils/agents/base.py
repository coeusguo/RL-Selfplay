import abc
import torch
import numpy as np
from typing import Tuple

class Agent(abc.ABC):

    def __init__(self, args, policy, game):
        self.policy = policy
        self.game = game
        self.args = args

    @abc.abstractmethod
    def init_buffer(cls, *args, **kwargs):
        '''
        generic methods, will be called by rollout sampler to clean
        the buffer (if any), after loading the new policy checkpoint
        '''
        pass

    @abc.abstractmethod
    def action_probs(self, canonical_board: np.ndarray, temperature: float, **kwargs):
        pass

    def run_one_episode(self, non_draw = False) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        self.policy.eval()

        # sample buffers
        raw_canonical_boards = []
        action_probs = []

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

            pi = self.action_probs(canonical_board, temperature=temp)

            # save samples
            raw_canonical_boards.append(canonical_board)
            action_probs.append(pi)

            # randomly sample an action from categorical distribution
            action_id = np.random.choice(np.arange(len(pi)), p=pi).item()

            # get updated board and opponent's id
            board, player_id = self.game.get_next_state(board, player_id, action_id)
            
            # check terminal condition
            reward = self.game.is_terminal(board, init_player_id)
            if non_draw and reward == 0:
                # draw, but requires non-draw game, restart the game
                board = self.game.get_empty_board()
                raw_canonical_boards = []
                action_probs = []
                player_id = init_player_id
                continue

            if reward is not None:
                # print(self.game.get_readable_board(board))
                # print(f"{reward=}")
                break

        return self.pack_samples([raw_canonical_boards], [action_probs], [reward], [init_player_id])[0]

    def pack_samples(self, 
            canonical_boards: list[list[np.ndarray]], 
            action_probs: list[list[np.ndarray]], 
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
        assert len(canonical_boards) == len(action_probs) == len(rewards) == len(init_player_ids)

        episode_lengths = []
        _boards, _action_probs, _player_ids, _rewards = [], [], [], []
        for idx in range(len(action_probs)):
            # note length for each episode
            elength = len(action_probs[idx])
            episode_lengths.append(elength)

            _boards.append(np.stack(canonical_boards[idx])) # [(b1, n, n), (b2, n, n), ...]
            _action_probs.append(np.stack(action_probs[idx])) # [(b1, n*n), (b2, n*n), ...]

            player_ids = np.ones(elength, dtype=np.int64) * init_player_ids[idx]
            player_ids[1::2] = - init_player_ids[idx]
            _player_ids.append(player_ids)

            _rewards.append(np.ones(elength, dtype=np.int64) * rewards[idx])
        
        # B = b1 + b2 + ... + b_n, where b_i is the length of i-th episode
        boards = np.concatenate(_boards, axis=0) #(B, N, N)
        action_probs = np.concatenate(_action_probs, axis=0) #(B, n*n)
        player_ids = np.concatenate(_player_ids, axis=0) #(B,)
        rewards = np.concatenate(_rewards, axis=0) #(B,)

        boards = torch.from_numpy(boards).float()
        action_probs = torch.from_numpy(action_probs)
        if torch.cuda.is_available():
            boards = boards.cuda()
            action_probs = action_probs.cuda()

        # do some data augmentation
        augmented_samples = self.game.get_symmetries(boards, action_probs) # (8, B, N*N)

        break_points = np.cumsum(episode_lengths)[:-1]
        group_rewards = rewards * player_ids
        group_rewards = list(np.split(group_rewards, break_points))
        group_rewards = [list(rewards) for rewards in group_rewards] 

        samples = [[] for _ in range(len(episode_lengths))]

        for (board, action_prob) in augmented_samples:
            board = board.cpu().numpy()
            action_prob = action_prob.cpu().numpy()

            board = list(np.split(board, break_points))
            action_prob = list(np.split(action_prob, break_points))

            for idx in range(len(board)):
                board_ls, pi_ls = list(board[idx]), list(action_prob[idx])
                samples[idx].extend(list(zip(board_ls, pi_ls, group_rewards[idx])))

        return samples

