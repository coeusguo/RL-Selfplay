import os
import abc
import random
import torch
import numpy as np
from tqdm import tqdm
from collections import deque, OrderedDict
from typing import Tuple
from pathlib import Path, PosixPath
from collections import defaultdict
from utils.optimizer import get_optimizer
from utils.models import get_wrapped_policy
from utils.losses import get_loss
from utils.games import get_game
from utils.agents import get_agent
from utils.games.arena import Arena


class BaseSelfPlayActor:
    def __init__(self, args):
        self.temp = args.temperature

        self.policy = get_wrapped_policy(args)
        self.policy.to(self.device)

        self.game = get_game(args)
        self.agent = get_agent(args, self.policy, self.game)

    def generate_one_episode(self, non_draw = False) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self.agent.run_one_episode(non_draw=non_draw)
        
    def load_checkpoint(self, state_dict: OrderedDict):
        self.policy.load_state_dict(state_dict)

        # the policy has been updated, should clear old buffers (if any)
        self.agent.init_buffer()


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        
        self.buffer_size = args.trainer_data_buffer_size
        self.num_mini_epoch = args.num_mini_epoch
        self.batch_size = args.batch_size
        self.sample_buffer = deque()

        # initialize policy model
        self.policy = get_wrapped_policy(args)
        self.policy.to(self.device)
        if args.ckpt is not None:
            self.load_checkpoint(args.ckpt)

        # save a game and agent object for self play matching
        self.game = get_game(self.args)
        self.agent = get_agent(self.args, self.policy, self.game)

        # initialize optimizer 
        self.optimizer = get_optimizer(args, self.policy.get_policy())
        
        # get loss function
        self.loss_fn = get_loss(args)

        # maintain a old policy for evaluation
        self.old_policy = get_wrapped_policy(self.args)
        self.old_policy.to(self.device)
        self.old_agent = get_agent(self.args, self.old_policy, self.game)
        self.update_old_policy()

    def np_to_tensor(self, array: list[np.ndarray]) -> torch.Tensor:
        array = np.array(array)
        array = torch.tensor(array, dtype=torch.float32)
        return array.to(self.device)

    def save_checkpoint(self, save_dir: PosixPath | str):

        state_dict = self.policy.state_dict()
        save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # could be used to resume training
        torch.save(state_dict, save_dir / "policy.pth")
        torch.save(self.optimizer.state_dict, save_dir / "optimizer.pth")

        return state_dict
    
    def get_checkpoint(self):
        return self.policy.state_dict()

    def load_checkpoint(self, ckpt_dir: str | PosixPath):
        '''
        this function is only used for resume training
        '''
        assert os.path.isdir(ckpt_dir), f"{ckpt_dir} does not exist"
        ckpt_dir = Path(ckpt_dir)
        policy_state_dict = torch.load(ckpt_dir / "policy.pth")
        self.policy.load_state_dict(policy_state_dict)

        # load optimizer state dict if exists
        if self.args.resume:
            assert os.path.exists(ckpt_dir / "optimizer.pth"), \
                "resume flag on, but no optimizer state_dict found"
            optimizer_state_dict = torch.load(ckpt_dir / "optimizer.pth")
            self.optimizer.load_state_dict(optimizer_state_dict)

    def update_train_sample(self, episode: list[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        if len(self.sample_buffer) > self.buffer_size:
            self.sample_buffer.popleft()
        self.sample_buffer.append(episode)

    def update_old_policy(self):
        state_dict = self.policy.state_dict()
        self.old_policy.load_state_dict(state_dict)

    def match_one_round(self, clear_cache_after_eval: bool = False):

        self.old_policy.eval()
        self.policy.eval()

        num_win, num_lose, num_draw, raw_samples = Arena.multiple_matches(
            self.agent,
            self.old_agent,
            self.game,
            num_match=2,
            use_tqdm=False,
            keep_samples=True,
            non_draw=True,
        )
        '''
            reuse the match samples for training
            raw_samples: [
                {
                "samples": [(board, action_id), (board, action_id), ...],
                "init_player_id": int
                "reward": int
                },
                ...
            ]
        '''
        if self.args.reuse_eval_sample:
            if len(raw_samples) > 0:
                boards, action_ids, rewards, init_player_ids = [], [], [], []
                for idx in range(len(raw_samples)):
                    board, action_id = zip(*raw_samples[idx]["samples"])
                    boards.append(board)
                    action_ids.append(action_id)
                    rewards.append(raw_samples[idx]["reward"])
                    init_player_ids.append(raw_samples[idx]["init_player_id"])
            
                samples = self.agent.pack_samples(boards, action_ids, rewards, init_player_ids)
                for sample in samples:
                    self.update_train_sample(sample)

        if clear_cache_after_eval:
            # since the new policy model will be updated, clear all the old buffers
            self.agent.init_buffer()
            self.old_agent.init_buffer()

        return num_win, num_lose, num_draw

    @abc.abstractmethod
    def train_one_epoch(self):
        pass


class MCTSSelfPlayActor(BaseSelfPlayActor):
    def __init__(self, args):
        super(MCTSSelfPlayActor, self).__init__(args)

        print(f"set the rollout search depth to {int(args.rollout_num_tree_search)}!")
        self.agent.mcts.num_tree_search = int(args.rollout_num_tree_search)


class MCTSTrainer(BaseTrainer):
    def __init__(self, args):
        super(MCTSTrainer, self).__init__(args)

    def train_one_epoch(self):
        self.policy.train()
        
        all_samples = []
        for episode in self.sample_buffer:
            all_samples.extend(episode)

        num_batches = len(all_samples) // self.batch_size
        assert num_batches > 0, f"too few training samples, only {len(all_samples)}"

        # shuffle the training sample
        random.shuffle(all_samples)
        
        print_dict = defaultdict(float)
        for e in range(self.num_mini_epoch):
            for idx in range(num_batches):
                start_idx = idx * self.batch_size
                end_idx = (idx + 1) * self.batch_size

                boards, target_pi, target_v = list(zip(*all_samples[start_idx: end_idx]))

                # convert to cuda tensor
                boards = self.np_to_tensor(boards) 
                target_pi = self.np_to_tensor(target_pi)
                target_v = self.np_to_tensor(target_v)

                log_pi, v = self.policy(boards)
                loss = self.loss_fn(log_pi, v, target_pi, target_v)

                for k, v in loss.items():
                    print_dict[k] += v.detach().cpu().item()

                self.optimizer.zero_grad()
                loss["loss"].backward()
                self.optimizer.step()

        return {
            k: f"{v / self.num_mini_epoch / num_batches:.3f}" for k, v in print_dict.items()
        }


class PPOTrainer(BaseTrainer):
    def __init__(self, args):
        super(PPOTrainer, self).__init__(args)

    def train_one_epoch(self, num_epoch: int | None = None):
        self.policy.train()

        print_dict = defaultdict(float)
        for _ in range(self.num_mini_epoch):
            for episode in self.sample_buffer:
                boards, old_pi, target_v = list(zip(*episode))

                # convert to cuda tensor
                boards = self.np_to_tensor(boards) 
                old_pi = self.np_to_tensor(old_pi)
                target_v = self.np_to_tensor(target_v)

                log_pi, v = self.policy(boards)
                loss_dict = self.loss_fn(self.args, log_pi, old_pi, v, target_v)
                loss = loss_dict.pop("loss")

                for k, v in loss.items():
                    print_dict[k] += v.detach().cpu().item()

                self.optimizer.zero_grad()
                loss["loss"].backward()
                self.optimizer.step()

        return {
            k: f"{v / self.num_mini_epoch / num_batches:.3f}" for k, v in print_dict.items()
        }
        