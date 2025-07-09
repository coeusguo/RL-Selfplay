import ray
from argument import parse_args
from utils.ray_actors import SelfPlayActor, Trainer
from collections import deque

def train(args):
    rollout_actor = SelfPlayActor(args)
    trainer = Trainer(args) 

    # deploy ray actors
    ray.get([rollout_actor.ready.remote(), trainer.ready.remote()])

    # rollout buffer
    rollout_buffer = deque()
    rollout_buffer_size = args.rollout_buffer_size

    # prefill the rollout_buffer
    while len(rollout_buffer) < rollout_buffer_size:
        rollout_buffer.append(rollout_actor.generate_samples.remote())

    epoch = 0
    # generate warm up examples
    while True:
        # genererate new samples if needed
        if len(rollout_buffer) < rollout_buffer_size:
            rollout_buffer.append(rollout_actor.generate_samples.remote())

    
        

if __name__ == "__main__":
    args = parse_args()
    ray.init(num_gpus=2)
    train(args)