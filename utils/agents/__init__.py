from .mcts import MCTSAgent

def get_agent(args, policy, game):
    if args.agent.lower() == "mcts":
        return MCTSAgent(args, policy, game)

    else:
        raise NotImplementedError(f"agent: {args.agent} not implemented")


if __name__ == "__main__":
    from argument import parse_args
    from utils.models import get_wrapped_policy
    from utils.games import get_game
    from tqdm import tqdm

    args = parse_args()

    game = get_game(args)
    policy = get_wrapped_policy(args).cuda()
    agent = MCTSAgent(args, policy, game)
    pb = tqdm(range(100))
    for _ in pb:
        agent.run_one_episode()
