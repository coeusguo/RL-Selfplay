import os
import ray
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import deque, OrderedDict
from typing import Literal, Tuple
from pathlib import Path, PosixPath
from .optimizer import get_optimizer
from .models import get_wrapped_policy
from .losses import get_loss
from .games import get_game
from .agents import get_agent
from .games.arena import Arena

@ray.remote(num_gpus=1)
class SelfPlayActor:
    def __init__(self, args):
        self.temp = args.temperature

        self.policy = get_wrapped_policy(args)
        self.game = get_game(args)
        self.agent = get_agent(args, self.policy, self.game)

        # setup ray gpu environment
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        self.device = torch.device("cuda:0")

        self.policy.to(self.device)

        print("hard coded the search depth to 20!")
        self.agent.mcts.num_tree_search = 20

    def ready(self):
        return True

    def generate_one_episode(self, non_draw = False) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self.agent.run_one_episode(non_draw=non_draw)
        
    def load_checkpoint(self, state_dict: OrderedDict):
        self.policy.load_state_dict(state_dict)

        # the policy has been updated, should clear old buffers (if any)
        self.agent.init_buffer(num_visits_only=False)

    def get_checkpoint(self):
        return self.policy.state_dict()


@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, args):
        self.args = args
        
        self.buffer_size = args.trainer_data_buffer_size
        self.num_mini_epoch = args.num_mini_epoch
        self.batch_size = args.batch_size
        self.sample_buffer = deque()

        # initialize policy model
        self.policy = get_wrapped_policy(args)
        if args.ckpt is not None:
            self.load_checkpoint(args.ckpt)

        # save a game and agent object for self play matching
        self.game = get_game(self.args)
        self.agent = get_agent(self.args, self.policy, self.game)

        # initialize optimizer 
        self.optimizer = get_optimizer(args, self.policy.get_policy())
        
        # get loss function
        self.loss_fn = get_loss(args)

        # setup ray gpu environment
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        self.device = torch.device("cuda:0")

        self.policy.to(self.device)

        # will store old policy
        self.old_agent = None
        
    def ready(self):
        return True

    def train_one_epoch(self):
        self.policy.train()

        def np_to_tensor(array: list[np.ndarray]) -> torch.Tensor:
            array = np.array(array)
            array = torch.tensor(array, dtype=torch.float32)
            return array.to(self.device)
        
        all_samples = []
        for episode in self.sample_buffer:
            all_samples.extend(episode)

        num_batches = len(all_samples) // self.batch_size
        assert num_batches > 0, f"too few training samples, only {len(all_samples)}"

        total_loss = 0
        for e in range(self.num_mini_epoch):
            # shuffle the training sample
            random.shuffle(all_samples)

            for idx in range(num_batches):
                start_idx = idx * self.batch_size
                end_idx = (idx + 1) * self.batch_size

                boards, target_pi, target_v = list(zip(*all_samples[start_idx: end_idx]))
                boards, target_pi, target_v = np_to_tensor(boards), np_to_tensor(target_pi), np_to_tensor(target_v)

                log_pi, v = self.policy(boards)
                loss = self.loss_fn(log_pi, v, target_pi, target_v)

                total_loss += loss.detach().cpu().item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return total_loss / self.num_mini_epoch / num_batches

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

    def match_one_round(self, 
        old_policy_ckpt: OrderedDict | None = None, 
        clear_cache_after_eval: bool = False
    ):
        if old_policy_ckpt is not None:
            if self.old_agent is None:
                self.old_policy = get_wrapped_policy(self.args)
                self.old_policy.to(self.device)
                self.old_agent = get_agent(self.args, self.old_policy, self.game)

            self.old_policy.load_state_dict(old_policy_ckpt)

        self.old_policy.eval()
        self.policy.eval()

        num_win, num_lose, num_draw, raw_samples = Arena.multiple_matches(
            self.agent.mcts,
            self.old_agent.mcts,
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

    def dual(self,):
        pass

@ray.remote(num_gpus=1)
class AyncMCTSOpponent:
    def __init__(self, args, player_id = -1):
        super(AyncMCTSAgent, self).__init__(args, policy, game)

        # setup ray gpu environment
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        self.device = torch.device("cuda:0")

        policy = get_wrapped_policy(args)
        try:
            state_dict = torch.load(Path(args.ckpt) / "policy.pth")
            policy.load_state_dict(state_dict)
        except:
            print("checking args.ckpt, but no checkpoint provided")

        policy.to(self.device)
        self.game = get_game(args)
        self.mcts = get_agent(args, policy, self.game).mcts

        # controls the endless searching loop
        self.stop_search = False

    def ready(self):
        return True
    
    def endless_tree_search(self, canonical_board):
        self.stop_search = False
        self.search_stopped = False
        while not self.stop_search:
            self.mcts.tree_search(canonical_board)
        self.search_stopped = True

    def stop_tree_search(self):
        self.stop_search = True
        while not self.search_stopped:
            continue
        return True

    def move(self, board):
        canonical_board = self.game.get_canonical_form(board, self.player_id)
        self.stop_tree_search()

        # perform tree search again
        for _ in range(self.mcts.num_tree_search):
            self.mcts.tree_search(canonical_board)

        action_id = np.argmax(self.mcts.action_probs(canonical_board, temperature=0))

        board, next_player_id = self.game.get_next_state(board, self.player_id, action_id)
        canonical_board = self.game.get_canonical_form(board, next_player_id)


        return action_id