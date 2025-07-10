import os
import ray
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import deque
from typing import Literal, Tuple
from pathlib import Path, PosixPath
from .optimizer import get_optimizer
from .models import get_wrapped_policy
from .losses import get_loss
from .games import get_game
from .agents import get_agent

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

    def ready(self):
        return True

    def generate_samples(self, num_episodes: int = 1 
                         ) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        samples = self.agent.run_one_episode()
        if num_episodes > 1:
            samples = [samples]
            for _ in range(1, num_episodes):
                samples.append(self.agent.run_one_episode())
        
        return samples
        
    def load_checkpoint(self, ckpt_dir: str | PosixPath):
        '''
        self play actor and trainer are pairs, must be created together
        should call Traininer.save_chekpoint before calling this function
        '''
        ckpt_dir = Path(ckpt_dir)
        assert os.path.isdir(ckpt_dir), f"{ckpt_dir} not found"

        state_dict = torch.load(ckpt_dir / "temp.pth")
        self.policy.load_state_dict(state_dict)
    

@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, args):
        self.buffer_size = args.sample_buffer_size
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.sample_buffer = deque()
        
        # initialize policy model
        self.policy = get_wrapped_policy(args).get_policy

        # initialize optimizer 
        self.optimizer = get_optimizer(args, self.agent.get_policy_net())
        
        # get loss function
        self.loss_fn = get_loss(args)

        # setup ray gpu environment
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        self.device = torch.device("cuda:0")
        
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
        for e in range(self.num_epoch):
            # shuffle the training sample
            random.shuffle(all_samples)

            progress_bar = tqdm(num_batches, desc=f"Training policy net, epoch[{e}/{self.num_epoch}]")
            for idx in range(num_batches):
                start_idx = idx * self.batch_size
                end_idx = (idx + 1) * self.batch_size

                boards, target_pi, target_v = list(zip(*all_samples[start_idx: end_idx]))
                boards, pi, v = np_to_tensor(boards), np_to_tensor(pi), np_to_tensor(v)

                pi, v = self.policy(boards)
                loss = self.loss_fn(pi, v, target_pi, target_v)

                progress_bar.set_postfix(loss=loss.detach().cpu().item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save_checkpoint(self, save_dir: str | PosixPath, load_by_rollout: bool = True):
        save_dir = Path(save_dir)
        os.mkdir(save_dir, exist_ok=True)
        state_dict = self.policy.state_dict()
        if load_by_rollout:
            # ckpt will be used to update rollout actor
            torch.save(state_dict, save_dir / "temp.pth")
        else:
            # could be used to resume training
            torch.save(state_dict, save_dir / "policy.pth")
            torch.save(self.optimizer.state_dict, save_dir / "optimizer.pth")

    def load_checkpoint(self, ckpt_dir: str | PosixPath):
        '''
        this function is only used for resume training
        '''
        assert os.path.isdir(ckpt_path), f"{ckpt_path} does not exist"
        ckpt_dir = Path(ckpt_dir)
        policy_state_dict = torch.load(ckpt_dir / "policy.pth")
        self.policy.load_state_dict(policy_state_dict)

        # load optimizer state dict if exists
        if os.path.exists(ckpt_dir / "optimizer.pth"):
            optimizer_state_dict = torch.load(ckpt_dir / "optimizer.pth")
            self.optimizer.load_state_dict(optimizer_state_dict)

    def update_train_sample(self, episodes):
        for s in episodes:
            if len(self.sample_buffer) > self.buffer_size:
                self.sample_buffer.popleft()
            self.sample_buffer.append(s)
