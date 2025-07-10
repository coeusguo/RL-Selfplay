import ray
from argument import parse_args
from utils.ray_actors import SelfPlayActor, Trainer
from collections import deque

def train(args):
    rollout_actor = SelfPlayActor(args)
    trainer = Trainer(args) 

    # deploy ray actors
    ray.get([rollout_actor.ready.remote(), trainer.ready.remote()])

    # sync initial model weights
    ray.get(trainer.save_checkpoint.remote(args.log_dir))
    ray.get(rollout_actor.load_checkpoint.remote(args.log_dir))

    # rollout buffer
    rollout_buffer = deque()
    rollout_buffer_size = args.rollout_buffer_size

    # prefill the traing sample
    rollout_ref = rollout_actor.generate_samples.remote(num_episodes=args.data_buffer_size)
    samples = ray.get(rollout_ref)
    trainer.update_train_sample.remote(samples)

    train_ref = trainer.train_one_epoch.remote()

    epoch = 0
    # generate warm up examples
    while True:
        # genererate new samples if needed
        if len(rollout_buffer) < rollout_buffer_size:
            rollout_buffer.append(rollout_actor.generate_samples.remote())

        ready_refs, _ = ray.wait([train_ref], timeout=1e-4)
        if len(ready_refs) > 0:
            # training done
            epoch += 1
            
            # check evaluate condition
            if epoch % args.eval_every:
                # evaluate, todo
                pass

                # update rollout policy if win rate reaches threshold

            # update training buffer
            if len(rollout_buffer) > 0:
                ready_refs, _ = ray.wait(rollout_buffer, num_returns=1)
                for _ in range(len(ready_refs)):
                    sample_ref = rollout_buffer.popleft()
                    samples = ray.get(sample_ref)
                    trainer.update_train_sample.remote(samples)

            train_ref = trainer.train_one_epoch.remote()
            
if __name__ == "__main__":
    args = parse_args()
    ray.init(num_gpus=2)
    train(args)