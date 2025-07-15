import ray
from .base_actors import MCTSSelfPlayActor, PPOTrainer, MCTSTrainer
from .ray_actors import RayMCTSSelfPlayActor, RayMCTSTrainer

def get_rollout_actor(args):
    if args.rollout_actor.lower() == "mcts":
        return MCTSSelfPlayActor(args)
    elif args.rollout_actor.lower() == "mcts_ray":
        rollout_actor = RayMCTSSelfPlayActor.remote(args)
        ray.get(rollout_actor.ready.remote())
        return rollout_actor
    else:
        raise NotImplementedError(f"unknown rollout actor: {args.rollout_actor}")


def get_trainer(args):
    if args.trainer.lower() == "mcts":
        return MCTSTrainer(args)
    elif args.trainer.lower() == "mcts_ray":
        trainer = RayMCTSTrainer.remote(args)
        ray.get(trainer.ready.remote())
        return trainer
    elif args.trainer.lower() == "ppo":
        return PPOTrainer(args)
    else:
        raise NotImplementedError(f"unknown trainer : {args.trainer}")
