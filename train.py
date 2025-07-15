from argument import parse_args
from utils.actors import get_trainer, get_rollout_actor

def train(args):
    rollout_actor = get_rollout_actor(args)
    trainer = get_trainer(args)

    epoch = 0
    while True:
        samples = []
        for _ in range(args.rollout_buffer_size):
            samples.append(rollout_actor)

        trainer.update_train_sample(samples)

        trainer.train_one_epoch()

if __name__ == "__main__":
    args = parse_args()
    train(args)