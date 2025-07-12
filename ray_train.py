import ray
from argument import parse_args
from utils.ray_actors import SelfPlayActor, Trainer
from collections import deque
from tqdm import tqdm

def train(args):
    rollout_actor = SelfPlayActor.remote(args)
    trainer = Trainer.remote(args) 

    # deploy ray actors
    ray.get([rollout_actor.ready.remote(), trainer.ready.remote()])

    # sync initial model weights
    state_dict = ray.get(trainer.get_checkpoint.remote())
    rollout_actor.load_checkpoint.remote(state_dict)

    # rollout buffer
    rollout_buffer = deque()
    rollout_buffer_size = args.rollout_buffer_size

    # prefill some traing samples
    temp_buffer = []
    temp_buffer = deque([rollout_actor.generate_one_episode.remote(non_draw=True) for _ in range(10)])
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
            loss = ray.get(train_ref)
            progress_bar.set_postfix(loss=f"{loss:.3f}")
            progress_bar.update(1)
            # training done
            epoch += 1
            
            # check evaluate condition
            if epoch % args.eval_every == 0:
                # evaluate
                old_state_dict = ray.get(rollout_actor.get_checkpoint.remote())
                match_refs = [trainer.match_one_round.remote(old_state_dict)]
                if args.num_eval_matches == 1:
                    match_refs = [trainer.match_one_round.remote(old_state_dict, clear_cache_after_eval=True)]
                elif args.num_eval_matches == 2:
                    match_refs = [
                        trainer.match_one_round.remote(old_state_dict),
                        trainer.match_one_round.remote(clear_cache_after_eval=True)
                        ]
                else:
                    match_refs = [trainer.match_one_round.remote(old_state_dict)] + \
                                 [trainer.match_one_round.remote() for _ in range(args.num_eval_matches - 2)] + \
                                 [trainer.match_one_round.remote(clear_cache_after_eval=True)] 
                
                eval_pb = tqdm(range(len(match_refs)), 
                               desc=f"Evaluation (old model: Gen {num_rollout_policy_update})", 
                               position=1, leave=False)
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

                win_rate = total_win / (total_win + total_lose)

                # update rollout policy if win rate reaches threshold
                if win_rate + 1e-5 > args.update_threshold:
                    num_rollout_policy_update += 1
                    state_dict = ray.get(trainer.get_checkpoint.remote())
                    rollout_actor.load_checkpoint.remote(state_dict)

            if epoch % args.save_every == 0:
                trainer.save_checkpoint.remote(args.log_dir)

            # update training buffer
            if len(rollout_buffer) > 0:
                ready_refs, _ = ray.wait(list(rollout_buffer))
                for _ in range(len(ready_refs)):
                    sample_ref = rollout_buffer.popleft()
                    samples = ray.get(sample_ref)
                    trainer.update_train_sample.remote(samples)

            train_ref = trainer.train_one_epoch.remote()

        if epoch >= args.main_epoch:
            break
            
if __name__ == "__main__":
    args = parse_args()
    ray.init(num_gpus=2)
    train(args)