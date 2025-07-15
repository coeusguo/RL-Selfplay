from .base import Agent
from .mcts import MCTS

class PPOAgent(Agent):
    def __init__(self, args, policy, game):
        super(PPOAgent, self).__init__(args, policy, game)
        self.mcts = None
        if getattr(self.args, "ppo_using_mcts", False):
            self.mcts = MCTS(self.args, self.game, self.policy)

    def init_buffer(self):
        if self.mcts is not None:
            self.mcts.clear_search_tree()
        
    def action_probs(self, canonical_board, temperature, do_search = True):
        if self.mcts is not None:
            return self.mcts(canonical_board, temperature, do_search)
        
        else:
            probs, _ = self.policy.predict(canonical_board)
            return probs
