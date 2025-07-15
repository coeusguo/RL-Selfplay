import os
import ray
import torch
from .base_actors import MCTSSelfPlayActor, MCTSTrainer


@ray.remote(num_gpus=1)
class RayMCTSSelfPlayActor(MCTSSelfPlayActor):
    def __init__(self, args):

        # setup ray gpu environment
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        self.device = torch.device("cuda:0")

        super().__init__(args)

    def ready(self):
        return True


@ray.remote(num_gpus=1) 
class RayMCTSTrainer(MCTSTrainer):
    def __init__(self, args):

        # setup ray gpu environment
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        self.device = torch.device("cuda:0")

        super().__init__(args)

    def ready(self):
        return True