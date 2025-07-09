from .mcts import MCTSAgent

def get_agent(args, policy, game):
    if args.agent.lower() == "mcts":
        return MCTSAgent(args, policy, game)

    else:
        raise NotImplementedError(f"agent: {args.agent} not implemented")