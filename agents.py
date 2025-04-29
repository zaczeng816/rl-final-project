import numpy as np
import yaml
import torch
from alpha_net_c4 import ConnectNet
from MCTS import UCT_search, do_decode_n_move_pieces, get_policy
from connect_board import Board 

class RandomAgent:
    def __init__(self):
        pass

    def play(self, env):
        action_space = env.action_space
        return action_space.sample()

class SmartAgent():
    
    def __init__(self):
        pass
    
    def play_single(self, observation):
        moves_to_avoid = []

        # Check for a winning move for the player
        for move in range(7):
            if observation[0, move] == 0:
                new_board, row, col = self.apply_move(observation, move, 1)
                if self.check_win_around_last_move(new_board, 1, row, col):
                    return move

        # Check for a winning move for the opponent
        for move in range(7):
            if observation[0, move] == 0:
                new_board, row, col = self.apply_move(observation, move, -1)
                if self.check_win_around_last_move(new_board, -1, row, col):
                    return move

        # Check if a move allows the opponent to win by playing the same move again
        for move in range(7):
            if observation[0, move] == 0:
                new_board, row, col = self.apply_move(observation, move, 1)
                if new_board[0, move] == 0:
                    new_board, row, col = self.apply_move(new_board, move, -1)
                    if self.check_win_around_last_move(new_board, -1, row, col):
                        moves_to_avoid.append(move)

        # Check if a move creates a line of three tokens with available spaces on both sides
        for move in range(7):
            if observation[0, move] == 0:
                new_board, row, col = self.apply_move(observation, move, 1)
                if self.check_three_in_a_row(new_board, 1, row, col) and move not in moves_to_avoid:
                    return move
        
        for move in range(7):
            if observation[0, move] == 0:
                new_board, row, col = self.apply_move(observation, move, -1)
                if self.check_three_in_a_row(new_board, -1, row, col) and move not in moves_to_avoid:
                    return move

        # Play a random move among valid moves, excluding moves_to_avoid
        valid_moves = [c for c in range(7) if observation[0, c] == 0 and c not in moves_to_avoid]

        # If valid_moves is empty, choose a random move among all possible moves
        if not valid_moves:
            valid_moves = [c for c in range(7) if observation[0, c] == 0]

        if not valid_moves:
            print('no valid_move.. i don t want to suicide myself',valid_moves,observation)
            exit()

        return np.random.choice(valid_moves)

    def play(self, env):
        board = env.board
        return self.play_single(board)
    
    def isDeterministic(self):
        return False

    def apply_move(self, board, move, player):
        new_board = board.copy()
        for i in range(5, -1, -1):
            if new_board[i, move] == 0:
                new_board[i, move] = player
                return new_board, i, move
        print('wtf')
        exit()

    def check_win_around_last_move(self, board, player, row, col):
        directions = [
            (1, 0),  # horizontal
            (0, 1),  # vertical
            (1, 1),  # diagonal /
            (1, -1)  # diagonal \
        ]

        for dr, dc in directions:
            count = 0
            for step in range(-3, 4):
                r, c = row + step * dr, col + step * dc
                if 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0

        return False

    def check_three_in_a_row(self, board, player, row, col):
        directions = [
            (1, 0),  # horizontal
            (0, 1),  # vertical
            (1, 1),  # diagonal /
            (1, -1)  # diagonal \
        ]

        for dr, dc in directions:
            count = 0
            for step in range(-2, 3):
                r, c = row + step * dr, col + step * dc
                if 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                    count += 1
                    if count == 3 and 0 <= r - dr < 6 and 0 <= c - dc < 7 and board[r - dr, c - dc] == 0 and 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7 and board[r + 3 * dr, c + 3 * dc] == 0:
                        return True
                else:
                    count = 0

        return False
    

class AlphaZeroAgent():
    def __init__(self, yaml_file, model_path):        
        configs = yaml.safe_load(open(yaml_file, 'r'))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ConnectNet(num_cols=configs['board']['num_cols'], num_rows=configs['board']['num_rows'], num_blocks=configs['model']['num_blocks'])
        model = model.to(device)
        model.eval()
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model
        self.device = device
        self.configs = configs


    def play(self, env):
        game_state = Board()
        for row in range(6):
            for col in range(7):
                if env.board[row, col] != 0:
                    game_state.current_board[row, col] = 'O' if env.board[row, col] == 1 else 'X'
        root = UCT_search(game_state, self.configs['mcts']['num_simulations'], self.model, None, self.device)
        policy = get_policy(root, temp=1)
        return np.random.choice(np.arange(game_state.num_cols), p = policy)