import numpy as np
from tqdm import tqdm

class Arena:
    
    @staticmethod
    def match(player1, player2, game, first_move = True):
        '''
        player 1: user (usually new policy), id: 1
        player 2: opponent (usually old policy), id: -1
        first_move: determine if player moves first
        '''
        players = {
            1: player1,
            -1: player2
        }
        board = game.get_empty_board()

        player_id = 1 if first_move else -1
        while True:
            cur_player = players[player_id]

            canonical_board = game.get_canonical_form(board, player_id)
            action_id = np.max(cur_player.action_prob(canonical_board, temperature=0))

            board, player_id = game.get_next_state(canonical_board, player_id, action_id)

            # check user winning condition
            reward = game.is_terminal(board, player_id=1)
            if reward is not None:
                return reward
            

    @staticmethod
    def multiple_matches(player1, player2, game, num_match: int, use_tqdm: bool = True):
        num_win = num_draw = num_lose = 0
        if use_tqdm:
            progress_bar = tqdm(num_match, desc="Arena Matching")

        for idx in range(num_match):
            first_move = True if idx % 2 == 0 else False
            reward = Arena.match(player1, player2, game, first_move)
            if reward == 1:
                num_win += 1
            elif reward == -1:
                num_lose += 1
            else:
                num_draw += 1
            

            if use_tqdm:
                win_rate = num_win / (idx + 1)
                lose_rate = num_lose / (idx + 1)
                draw_rate = 1 - win_rate - lose_rate

                progress_bar.set_postfix(
                    win_rate=f"{win_rate * 100:.1f}%", 
                    lose_rate=f"{lose_rate * 100:.1f}%", 
                    draw_rate=f"{draw_rate * 100:.1f}%")
                progress_bar.step()

        return num_win, num_lose, num_draw
        