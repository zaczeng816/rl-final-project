"""

Defines various Connect Four agents:
- RandomAgent: selects uniform random valid moves.
- HeuristicAgent: rule-based heuristic for win/block/threats.
- AlphaZeroAgent: MCTS-driven agent using a neural network.
- ChildPlayer, ChildSmarterPlayer: simple heuristic players.
- BabyPlayer, BabySmarterPlayer: random or basic heuristic players.

"""

import numpy as np
import yaml
import torch
from model import ConnectNet
from MCTS import UCT_search, get_policy
from connect_board import Board
from connect_four_env import ConnectFourEnv

class RandomAgent():
    def __init__(self):
        pass

    def play(self, env):
        action_space = env.action_space
        return action_space.sample()


class HeuristicAgent:
    def __init__(self):
        pass

    def play(self, env):
        board = env.board
        # dynamic dimensions and win condition from env
        self.rows, self.cols = board.shape
        self.win_length = env.WIN_LENGTH
        self.main_id = env.main_player_id
        self.opp_id = env.opponent_id
        return self.play_single(board)

    def play_single(self, board):
        moves_to_avoid = []

        for move in range(self.cols):
            if board[0, move] == 0:
                new_board, r, c = self.apply_move(board, move, self.main_id)
                if self.check_win(new_board, self.main_id, r, c):
                    return move

        for move in range(self.cols):
            if board[0, move] == 0:
                new_board, r, c = self.apply_move(board, move, self.opp_id)
                if self.check_win(new_board, self.opp_id, r, c):
                    return move

        for move in range(self.cols):
            if board[0, move] == 0:
                new_board, r, c = self.apply_move(board, move, self.main_id)
                if new_board[0, move] == 0:
                  new_board, r, c = self.apply_move(new_board, move, self.opp_id)
                  if self.check_win(new_board, self.opp_id, r, c):
                    moves_to_avoid.append(move)

        for move in range(self.cols):
            if board[0, move] == 0:
                new_board, r, c = self.apply_move(board, move, self.main_id)
                if self.check_in_a_row(new_board, self.main_id, r, c) and move not in moves_to_avoid:
                    return move

        for move in range(self.cols):
            if board[0, move] == 0:
                new_board, row, col = self.apply_move(board, move, self.opp_id)
                if self.check_in_a_row(new_board, self.opp_id, row, col) and move not in moves_to_avoid:
                    return move


        # Play a random move among valid moves, excluding moves_to_avoid
        valid_moves = [c for c in range(self.cols) if board[0, c] == 0 and c not in moves_to_avoid]

        # If valid_moves is empty, choose a random move among all possible moves
        if not valid_moves:
            valid_moves = [c for c in range(self.cols) if board[0, c] == 0]

        if not valid_moves:
            print('no valid_move.. i don t want to suicide myself',valid_moves,board)
            exit()

        return np.random.choice(valid_moves)

    def apply_move(self, board, move, player):
        new_board = board.copy()
        for i in range(self.rows - 1, -1, -1):
            if new_board[i, move] == 0:
                new_board[i, move] = player
                return new_board, i, move
        print('Invalid Move')
        exit()

    def check_win(self, board, player, row, col):
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in dirs:
            count = 0
            for offset in range(-(self.win_length-1), self.win_length):
                r = row + dr*offset
                c = col + dc*offset
                if 0 <= r < self.rows and 0 <= c < self.cols and board[r, c] == player:
                    count += 1
                    if count >= self.win_length:
                        return True
                else:
                    count = 0
        return False

    def check_in_a_row(self, board, player, row, col):
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in dirs:
            count = 0
            for offset in range(-(self.win_length-1), self.win_length):
                r = row + dr*offset
                c = col + dc*offset
                if 0 <= r < self.rows and 0 <= c < self.cols and board[r, c] == player:
                    count += 1
                else:
                    if count == self.win_length - 1:
                        before_r = r - dr
                        before_c = c - dc
                        after_r = row + dr*self.win_length
                        after_c = col + dc*self.win_length
                        if (0 <= before_r < self.rows and 0 <= before_c < self.cols and board[before_r, before_c] == 0 and
                            0 <= after_r  < self.rows and 0 <= after_c  < self.cols and board[after_r,  after_c]  == 0):
                            return True
                    count = 0
        return False
    

class AlphaZeroAgent():
    def __init__(self, yaml_file, model_path):        
        configs = yaml.safe_load(open(yaml_file, 'r'))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ConnectNet(num_cols=configs['board']['num_cols'], num_rows=configs['board']['num_rows'], num_blocks=configs['model']['num_blocks'])
        model = model.to(device)
        model.eval()
        checkpoint = torch.load(model_path, map_location=device)
        if model_path.endswith('.pth.tar'):
          model.load_state_dict(checkpoint['state_dict'])
        elif model_path.endswith('.pth'):
          model.load_state_dict(checkpoint)
        else:
          raise ValueError(f"Unrecognized model extension: {model_path}")
        self.model = model
        self.device = device
        self.configs = configs
        self.temperature = 0.1


    def play(self, env: ConnectFourEnv):
        R, C = env.ROWS, env.COLS
        W = env.WIN_LENGTH

        game_state = Board(num_rows=R, num_cols=C, win_streak=W)

        for r in range(R):
            for c in range(C):
                v = env.board[r, c]
                if   v == 1: game_state.current_board[r, c] = 'O'
                elif v == 2: game_state.current_board[r, c] = 'X'

        game_state.player = 0 if env.current_player == 1 else 1

        root   = UCT_search(
            game_state,
            self.configs['mcts']['num_simulations'],
            self.model,
            self.temperature,
            self.device
        )
        policy = get_policy(root, self.temperature)

        return int(np.random.choice(np.arange(C), p=policy))


class Player:
    def __init__(self, name):
        self.name = name

    def play(self, env):
        raise NotImplementedError("The 'play' method must be implemented in the child class")
    
    def getName(self):
        return self.name

    def isDeterministic(self):
        raise NotImplementedError("The 'isDeterministic' method must be implemented in the child class")


class ChildPlayer(Player):
    def __init__(self, _=""):
        pass

    def play(self, env: ConnectFourEnv):

        board      = env.board
        rows, cols = env.ROWS, env.COLS
        win_length = env.WIN_LENGTH

        return self.play_single(board, rows, cols, win_length)

    def play_single(self, board, rows, cols, win_length):
        for move in range(cols):
            if board[0, move] == 0:
                new_board, r, c = self.apply_move(board, move, 1, rows)
                if self.check_win(new_board, 1, r, c, rows, cols, win_length):
                    return move

        valid_moves = [c for c in range(cols) if board[0, c] == 0]
        return int(np.random.choice(valid_moves))

    def apply_move(self, board, move, player, rows):
        new_board = board.copy()
        for i in range(rows - 1, -1, -1):
            if new_board[i, move] == 0:
                new_board[i, move] = player
                return new_board, i, move

    def check_win(self, board, player, row, col, rows, cols, win_length):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 0
            for offset in range(-(win_length - 1), win_length):
                r = row + dr * offset
                c = col + dc * offset
                if 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                    count += 1
                    if count >= win_length:
                        return True
                else:
                    count = 0
        return False

    def getName(self):
        return "ChildPlayer"

    def isDeterministic(self):
        return False
    

class ChildSmarterPlayer(Player):

    def __init__(self, _=""):
        pass

    def play_single(self, board, rows, cols, win_length):
        for move in range(cols):
            if board[0, move] == 0:
                new_board, r, c = self.apply_move(board, move, 1, rows)
                if self.check_win(new_board, 1, r, c, rows, cols, win_length):
                    return move
                
        for move in range(cols):
            if board[0, move] == 0:
                new_board, r, c = self.apply_move(board, move, 2, rows)
                if self.check_win(new_board, 2, r, c, rows, cols, win_length):
                    return move

        valid_moves = [c for c in range(cols) if board[0, c] == 0]
        return int(np.random.choice(valid_moves))

    def play(self, env: ConnectFourEnv):

        board      = env.board
        rows, cols = env.ROWS, env.COLS
        win_length = env.WIN_LENGTH

        return self.play_single(board, rows, cols, win_length)

    def getName(self):
        return "ChildSmarterPlayer"

    def isDeterministic(self):
        return False

    def apply_move(self, board, move, player, rows):
        new_board = board.copy()
        for i in range(rows - 1, -1, -1):
            if new_board[i, move] == 0:
                new_board[i, move] = player
                return new_board, i, move

    def check_win(self, board, player, row, col, rows, cols, win_length):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 0
            for offset in range(-(win_length - 1), win_length):
                r = row + dr * offset
                c = col + dc * offset
                if 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                    count += 1
                    if count >= win_length:
                        return True
                else:
                    count = 0
        return False


class BabyPlayer(Player):

    def __init__(self, _=""):
        pass

    def random_move(self, cols):
        return np.random.randint(0, cols)

    def play(self, env: ConnectFourEnv):
        cols = env.COLS
        return self.random_move(cols)

    def getName(self):
        return "BabyPlayer"


    def isDeterministic(self):
        return False


class BabySmarterPlayer(Player):

    def __init__(self, _=""):
        pass

    def play_single(self,cols, board):
        valid_moves = [c for c in range(cols) if board[0, c] == 0]
        return np.random.choice(valid_moves)

    def play(self, env: ConnectFourEnv):
        return self.play_single(env.COLS, env.board)

    def getName(self):
        return "BabySmarterPlayer"
    
    def isDeterministic(self):
        return False


