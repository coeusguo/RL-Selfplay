import ray
import torch
from argument import parse_args
from utils.actors import get_rollout_actor, get_trainer
from collections import deque
from tqdm import tqdm
from pathlib import Path

def train(args):
    rollout_actor = get_rollout_actor(args)
    trainer = get_trainer(args) 

    # sync initial model weights
    state_dict = ray.get(trainer.get_checkpoint.remote())
    rollout_actor.load_checkpoint.remote(state_dict)

    # rollout buffer
    rollout_buffer = deque()
    rollout_buffer_size = args.rollout_buffer_size

    # prefill some traing samples
    temp_buffer = []
    temp_buffer = deque([rollout_actor.generate_one_episode.remote() \
                                for _ in range(min(10, args.trainer_data_buffer_size))])

    for _ in tqdm(range(len(temp_buffer)), desc="Generating Initial Train Samples"):
        samples = ray.get(temp_buffer.popleft())
        trainer.update_train_sample.remote(samples)

    train_ref = trainer.train_one_epoch.remote()

    epoch = 0
    # generate warm up examples
    progress_bar = tqdm(range(args.main_epoch), desc="Self-Play Training", position=0)
    num_rollout_policy_update = 0
    while True:
        # genererate new samples if needed
        if len(rollout_buffer) < rollout_buffer_size:
            rollout_buffer.append(rollout_actor.generate_one_episode.remote())
        
        ready_refs, _ = ray.wait([train_ref], timeout=1e-4)
        if len(ready_refs) > 0:
            print_dict = ray.get(train_ref)
            progress_bar.set_postfix(**print_dict)
            progress_bar.update(1)
            # training done
            epoch += 1
            
            # check evaluate condition
            if epoch % args.eval_every == 0:
                # evaluate
                if args.num_eval_matches == 1:
                    match_refs = [trainer.match_one_round.remote(clear_cache_after_eval=True)]
                else:
                    match_refs = [trainer.match_one_round.remote() for _ in range(args.num_eval_matches - 1)] + \
                                 [trainer.match_one_round.remote(clear_cache_after_eval=True)] 
                
                eval_pb = tqdm(range(len(match_refs)), 
                               desc=f"Evaluation (old model: Gen {num_rollout_policy_update})", 
                               position=1)
                total_win = total_lose = total_draw = total_matches = 0
                for idx in eval_pb:
                    num_win, num_lose, num_draw = ray.get(match_refs[idx])
                    total_win += num_win
                    total_lose += num_lose
                    total_draw += num_draw
                    total_matches += (num_win + num_lose + num_draw)
                    eval_pb.set_postfix(
                        draw_rate=f"{total_draw / total_matches * 100:.2f}%",
                        lose_rate=f"{total_lose / total_matches * 100:.2f}%",
                        win_rate=f"{total_win / total_matches * 100:.2f}%"
                        )

                win_rate = total_win / (total_win + total_lose + 1e-5)

                # update rollout policy if win rate reaches threshold
                if win_rate + 1e-5 > args.update_threshold:
                    print(f"better model found, beat previous model with winning rate: {win_rate * 100:.2f}%")
                    num_rollout_policy_update += 1

                    # update rollout policy
                    state_dict = ray.get(trainer.get_checkpoint.remote())

                    # cancel all the unfinished refs 
                    _, unready_refs = ray.wait(list(rollout_buffer), timeout=1e-4)
                    for _ in range(len(unready_refs)):
                        ray.cancel(rollout_buffer.pop(), force=False)
                    
                    # load new checkpoint
                    rollout_actor.load_checkpoint.remote(state_dict)

                    # save best ckpt
                    torch.save(state_dict, Path(args.log_dir) / "policy_best.pth")

                    # update old policy
                    trainer.update_old_policy.remote()

            # update training buffer
            if len(rollout_buffer) > 0:
                ready_refs, _ = ray.wait(list(rollout_buffer), timeout=1e-4)
                if len(ready_refs) == 0:
                    ready_refs, _ = ray.wait(list(rollout_buffer), num_returns=1)

                for _ in range(len(ready_refs)):
                    sample_ref = rollout_buffer.popleft()
                    samples = ray.get(sample_ref)
                    trainer.update_train_sample.remote(samples)
                    
            if epoch % args.save_every == 0:
                trainer.save_checkpoint.remote(args.log_dir)

            # start new training epoch
            train_ref = trainer.train_one_epoch.remote()

        if epoch >= args.main_epoch:
            break
            
if __name__ == "__main__":

    args = parse_args()
    ray.init(num_gpus=2)
    train(args)