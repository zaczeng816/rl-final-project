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
    """
    A simple agent that makes random moves from available actions.
    Used as a baseline opponent for evaluating AI performance.
    """
    def __init__(self, num_cols, num_rows):
        self.num_cols = num_cols
        self.num_rows = num_rows

    def play(self, board):
        available_actions = board.actions()
        return available_actions[np.random.randint(0, len(available_actions))]


def play_game(net, configs, device, ai_first: bool):
    """
    Simulates a single game between the AI (neural network) and a random agent.
    
    Args:
        net: The neural network model
        configs: Configuration dictionary
        device: Device to run computations on (CPU/GPU)
        ai_first: Boolean indicating if AI plays first (as black)
    
    Returns:
        True if AI wins, False if random agent wins, None if draw
    """
    white = None; black = None
    # Set up players based on who goes first
    if ai_first:
        black = net
        white = RandomAgent(configs['board']['num_cols'], configs['board']['num_rows'])
    else:
        white = net
        black = RandomAgent(configs['board']['num_cols'], configs['board']['num_rows'])
        
    # Initialize game board
    current_board = Board(num_cols=configs['board']['num_cols'], num_rows=configs['board']['num_rows'], win_streak=configs['board']['win_streak'])
    checkmate = False
    value = 0; t = 0.1; moves_count = 0
    
    # Main game loop
    while checkmate == False and current_board.actions() != []:
        moves_count += 1
        # White's turn (player 0)
        if current_board.player == 0:
            if isinstance(white, RandomAgent):
                col = white.play(current_board)
                policy = np.zeros([current_board.num_cols], dtype=np.float32)
                policy[col] += 1
            else:
                # AI's turn - use MCTS to determine best move
                root = UCT_search(current_board, configs['mcts']['num_simulations'], white, t, device)
                policy = get_policy(root, t)
        # Black's turn (player 1)
        elif current_board.player == 1:
            if isinstance(black, RandomAgent):
                col = black.play(current_board)
                policy = np.zeros([current_board.num_cols], dtype=np.float32)
                policy[col] += 1
            else:
                # AI's turn - use MCTS to determine best move
                root = UCT_search(current_board, configs['mcts']['num_simulations'], black, t, device)
                policy = get_policy(root, t)

        # Execute the selected move
        next_move = np.random.choice(np.arange(current_board.num_cols), p = policy)
        current_board.drop_piece(next_move)

        # Check if game is over
        if current_board.check_winner() == True: # someone wins
            if current_board.player == 0: # black wins
                value = -1
            elif current_board.player == 1: # white wins
                value = 1
            checkmate = True

    # Determine game result
    if value == -1:
        return ai_first  # AI wins if it's black and black won
    elif value == 1:
        return not ai_first  # AI wins if it's white and white won
    else:
        return None  # Draw


def evaluate_net(net, configs, device, num_games: int, result_queue):
    """
    Evaluates the neural network by playing multiple games against a random agent.
    
    Args:
        net: The neural network model
        configs: Configuration dictionary
        device: Device to run computations on (CPU/GPU)
        num_games: Number of games to play
        result_queue: Queue to store results for parallel processing
    """
    ai_first_wins = 0
    random_agent_wins = 0
    
    # Test AI performance when going first and second
    for ai_first in [True, False]:
        wins = 0
        for _ in trange(num_games):
            result = play_game(net, configs, device, ai_first=ai_first)
            if result:
                wins += 1

        # Track wins based on who went first
        if ai_first:
            ai_first_wins += wins
        else:
            random_agent_wins += wins

    # Send back the results through the queue
    result_queue.put((ai_first_wins, random_agent_wins))


def parallel_evaluate_net(connectnet, configs, device):
    """
    Evaluates the neural network using multiple parallel processes.
    
    Args:
        connectnet: The neural network model
        configs: Configuration dictionary
        device: Device to run computations on (CPU/GPU)
    
    Returns:
        Tuple of (AI winrate when going first, AI winrate when going second)
    """
    # Enable model sharing between processes
    connectnet.share_memory()
    mp.set_start_method('spawn', force=True)

    # Determine number of processes and games per process
    n_procs = configs['self_play']['MCTS_num_processes']
    games_per_proc = configs['self_play']['num_evaluator_games'] // n_procs

    # Set up multiprocessing
    result_queue = mp.Queue()
    procs = []
    for _ in range(n_procs):
        p = mp.Process(target=evaluate_net, args=(connectnet, configs, device, games_per_proc, result_queue))
        p.start()
        procs.append(p)

    # Wait for all processes to complete
    for p in procs:
        p.join()

    # Collect results from all processes
    total_ai_first_wins = 0
    total_random_agent_wins = 0
    while not result_queue.empty():
        ai_first_wins, random_agent_wins = result_queue.get()
        total_ai_first_wins += ai_first_wins
        total_random_agent_wins += random_agent_wins

    # Calculate winrates
    num_games = n_procs * games_per_proc
    winrate_ai_first = total_ai_first_wins / num_games
    winrate_random_agent = total_random_agent_wins / num_games

    return winrate_ai_first, winrate_random_agent

if __name__ == "__main__":
    # Load model and configurations
    best_net_filename = "model_ckpts/cc3_current_net_small_step28000.pth"
    configs = yaml.safe_load(open('configs/h5_w5_c3_small.yaml', 'r'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize neural network
    best_cnet = ConnectNet(num_cols=configs['board']['num_cols'], num_rows=configs['board']['num_rows'], num_blocks=configs['model']['num_blocks'])
    cuda = torch.cuda.is_available()
    if cuda:
        best_cnet.cuda()
    best_cnet.eval()
    
    # Load model weights
    checkpoint = torch.load(best_net_filename)
    best_cnet.load_state_dict(checkpoint)
    
    # Evaluate model performance
    stats = parallel_evaluate_net(best_cnet, configs, device)
    print("AI first winrate: ", stats[0])
    print("Random agent winrate: ", stats[1])