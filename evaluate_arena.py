#!/usr/bin/env python
import torch
import yaml
import numpy as np

from model import ConnectNet
from connect_board import Board
from MCTS import UCT_search, get_policy
from tqdm import trange
from torch import multiprocessing as mp


class RandomAgent:
    def __init__(self, num_cols, num_rows):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def play(self, board):
        available_actions = board.actions()
        return available_actions[np.random.randint(0, len(available_actions))]


def play_game(net, configs, device, ai_first: bool):
    white = None; black = None
    if ai_first:
        black = net
        white = RandomAgent(configs['board']['num_cols'], configs['board']['num_rows'])
    else:
        white = net
        black = RandomAgent(configs['board']['num_cols'], configs['board']['num_rows'])
        
    current_board = Board(num_cols=configs['board']['num_cols'], num_rows=configs['board']['num_rows'], win_streak=configs['board']['win_streak'])
    checkmate = False
    value = 0; t = 0.1; moves_count = 0
    while checkmate == False and current_board.actions() != []:
        moves_count += 1
        if current_board.player == 0:
            if isinstance(white, RandomAgent):
                col = white.play(current_board)
                policy = np.zeros([current_board.num_cols], dtype=np.float32)
                policy[col] += 1
            else:
                root = UCT_search(current_board, configs['mcts']['num_simulations'], white, t, device)
                policy = get_policy(root, t)
        elif current_board.player == 1:
            if isinstance(black, RandomAgent):
                col = black.play(current_board)
                policy = np.zeros([current_board.num_cols], dtype=np.float32)
                policy[col] += 1
            else:
                root = UCT_search(current_board, configs['mcts']['num_simulations'], black, t, device)
                policy = get_policy(root, t)

        next_move = np.random.choice(np.arange(current_board.num_cols), p = policy)
        current_board.drop_piece(next_move)

        if current_board.check_winner() == True: # someone wins
            if current_board.player == 0: # black wins
                value = -1
            elif current_board.player == 1: # white wins
                value = 1
            checkmate = True

    if value == -1:
        return ai_first  # AI wins
    elif value == 1:
        return not ai_first  # RandomAgent
    else:
        return None


def evaluate_net(net, configs, device, num_games: int, result_queue):
    ai_first_wins = 0
    random_agent_wins = 0
    for ai_first in [True, False]:
        wins = 0
        for _ in trange(num_games):
            result = play_game(net, configs, device, ai_first=ai_first)
            if result:
                wins += 1

        if ai_first:
            ai_first_wins += wins
        else:
            random_agent_wins += wins

    # Send back the results
    result_queue.put((ai_first_wins, random_agent_wins))


def parallel_evaluate_net(connectnet, configs, device):
    connectnet.share_memory()
    mp.set_start_method('spawn', force=True)

    n_procs = configs['self_play']['MCTS_num_processes']
    games_per_proc = configs['self_play']['num_evaluator_games'] // n_procs

    result_queue = mp.Queue()
    procs = []
    for _ in range(n_procs):
        p = mp.Process(target=evaluate_net, args=(connectnet, configs, device, games_per_proc, result_queue))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    total_ai_first_wins = 0
    total_random_agent_wins = 0
    while not result_queue.empty():
        ai_first_wins, random_agent_wins = result_queue.get()
        total_ai_first_wins += ai_first_wins
        total_random_agent_wins += random_agent_wins

    num_games = n_procs * games_per_proc
    winrate_ai_first = total_ai_first_wins / num_games
    winrate_random_agent = total_random_agent_wins / num_games

    return winrate_ai_first, winrate_random_agent

if __name__ == "__main__":
    best_net_filename = "model_ckpts/cc3_current_net_small_step28000.pth"
    configs = yaml.safe_load(open('configs/h5_w5_c3_small.yaml', 'r'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_cnet = ConnectNet(num_cols=configs['board']['num_cols'], num_rows=configs['board']['num_rows'], num_blocks=configs['model']['num_blocks'])
    cuda = torch.cuda.is_available()
    if cuda:
        best_cnet.cuda()
    best_cnet.eval()
    checkpoint = torch.load(best_net_filename)
    best_cnet.load_state_dict(checkpoint)
    stats = parallel_evaluate_net(best_cnet, configs, device)
    print("AI first winrate: ", stats[0])
    print("Random agent winrate: ", stats[1])