#!/usr/bin/env python
import os

# Limit the number of threads to avoid resource contention
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pickle
import collections
import numpy as np
import math
import copy
import torch
import torch.multiprocessing as mp
import datetime
import queue

from tqdm import tqdm
from loguru import logger

from connect_board import Board, encode_board

# Set up logging
logger.add("logs/mcts.log")

def save_as_pickle(filename, data):
    """Save data to a pickle file in the datasets directory"""
    completeName = os.path.join("./datasets/", filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(filename):
    """Load data from a pickle file in the datasets directory"""
    completeName = os.path.join("./datasets/", filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class UCTNode():
    """
    MCTS tree node implementation

    Adopted from https://github.com/plkmo/AlphaZero_Connect4/blob/01ff24aae145ccd23e58f630aa49582cade49847/src/MCTS_c4.py
    """
    def __init__(self, game, move, parent=None):
        self.game = game # state s
        self.move = move # action index
        self.is_expanded = False
        self.parent = parent  
        self.children = {}
        self.child_priors = np.zeros([game.num_cols], dtype=np.float32)
        self.child_total_value = np.zeros([game.num_cols], dtype=np.float32)
        self.child_number_visits = np.zeros([game.num_cols], dtype=np.float32)
        self.action_idxes = []
        
    @property
    def number_visits(self):
        """Get the number of visits for this node from parent"""
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        """Set the number of visits for this node in parent"""
        self.parent.child_number_visits[self.move] = value
    
    @property
    def total_value(self):
        """Get the total value for this node from parent"""
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value):
        """Set the total value for this node in parent"""
        self.parent.child_total_value[self.move] = value
    
    def child_Q(self):
        """Calculate Q values (exploitation term) for all children"""
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self, c_puct=1.0):
        """Calculate U values (exploration term) for all children"""
        return c_puct * self.child_priors * math.sqrt(self.number_visits) / (1 + self.child_number_visits)
    
    def best_child(self, c_puct=1.0):
        """Select the best child based on Q+U values"""
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U(c_puct)
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U(c_puct))
        return bestmove
    
    def select_leaf(self, c_puct=1.0):
        """
        Select a leaf node by traversing the tree using the PUCT algorithm
        until reaching an unexpanded node
        """
        current = self
        while current.is_expanded:
            best_move = current.best_child(c_puct)
            current = current.maybe_add_child(best_move)
        return current
    
    def add_dirichlet_noise(self,action_idxs,child_priors):
        """
        Add Dirichlet noise to the prior probabilities at the root node
        to encourage exploration
        """
        valid_child_priors = child_priors[action_idxs] # select only legal moves entries in child_priors array
        valid_child_priors = 0.75 * valid_child_priors \
            + 0.25 * np.random.dirichlet(
                np.zeros([len(valid_child_priors)], dtype=np.float32) + 0.03 * len(valid_child_priors)
            )
        child_priors[action_idxs] = valid_child_priors
        return child_priors
    
    def expand(self, child_priors):
        """
        Expand the node by setting prior probabilities for all possible actions
        from the current state
        """
        self.is_expanded = True
        action_idxs = self.game.actions()
        c_p = child_priors
        if action_idxs == []:
            self.is_expanded = False
            return
        self.action_idxes = action_idxs
        c_p[[i for i in range(len(child_priors)) if i not in action_idxs]] = 0.000000000 # mask all illegal actions
        if self.parent.parent == None: # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_idxs,c_p)
        self.child_priors = c_p
    
    def decode_n_move_pieces(self,board,move):
        """Execute a move on the board"""
        board.drop_piece(move)
        return board
            
    def maybe_add_child(self, move):
        """
        Add a child node for the given move if it doesn't exist yet,
        and return the child node
        """
        if move not in self.children:
            copy_board = copy.deepcopy(self.game) # make copy of board
            copy_board = self.decode_n_move_pieces(copy_board,move)
            self.children[move] = UCTNode(copy_board, move, parent=self)
        return self.children[move]
    
    def backup(self, value_estimate: float):
        """
        Update the node statistics (visits and values) by propagating
        the value estimate up the tree
        """
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1: # same as current.parent.game.player = 0
                current.total_value += (1*value_estimate) # value estimate +1 = O wins
            elif current.game.player == 0: # same as current.parent.game.player = 1
                current.total_value += (-1*value_estimate)
            current = current.parent


class DummyNode(object):
    """
    Dummy node to serve as the parent of the root node in the MCTS tree,
    storing visit counts and values
    """
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


class BatchedEvaluator:
    """
    Evaluates multiple leaf nodes in batches to improve efficiency
    when using neural networks
    """
    def __init__(self, net, device, batch_size=1):
        self.net = net
        self.device = device
        self.batch_size = batch_size
        self._buf_states = []
        self._buf_leaves = []

    def enqueue(self, leaf):
        """Add a leaf node to the evaluation queue"""
        s = encode_board(leaf.game)
        t = torch.from_numpy(s.transpose(2,0,1)).float()
        self._buf_states.append(t)
        self._buf_leaves.append(leaf)
        if len(self._buf_states) >= self.batch_size:
            self.flush()

    @torch.no_grad()
    def flush(self):
        """
        Process all queued leaf nodes by evaluating them with the neural network
        and updating the tree accordingly
        """
        if not self._buf_states: return
        batch = torch.stack(self._buf_states, dim=0).to(
            self.device, non_blocking=True
        )
        priors_batch, values_batch = self.net(batch)
        priors_batch = priors_batch.cpu().numpy()
        values_batch = values_batch.cpu().numpy()
        for leaf, ps, v in zip(self._buf_leaves, priors_batch, values_batch):
            if leaf.game.check_winner() or leaf.game.actions()==[]:
                leaf.backup(v.item())
            else:
                leaf.expand(ps.reshape(-1))
                leaf.backup(v.item())
        self._buf_states.clear()
        self._buf_leaves.clear()


def UCT_search(game_state, num_reads, net, temp, device, c_puct=1.0, batch_size=1):
    """
    Perform MCTS search from the given game state using the neural network
    for evaluation
    
    Args:
        game_state: Current state of the game
        num_reads: Number of MCTS simulations to run
        net: Neural network for state evaluation
        temp: Temperature parameter for exploration
        device: Device to run computations on (CPU/GPU)
        c_puct: Exploration constant for PUCT algorithm
        batch_size: Batch size for neural network evaluation
        
    Returns:
        Root node of the MCTS tree
    """
    root = UCTNode(game_state, move=None, parent=DummyNode())
    evaluator = BatchedEvaluator(net, device, batch_size=batch_size)
    for _ in range(num_reads):
        leaf = root.select_leaf(c_puct)
        evaluator.enqueue(leaf)
    evaluator.flush()
    return root


def get_policy(root, temp=1):
    """
    Extract the policy (move probabilities) from the visit counts at the root node,
    applying a temperature parameter to control exploration vs. exploitation
    """
    return ((root.child_number_visits)**(1/temp))/sum(root.child_number_visits**(1/temp))


@torch.no_grad()
def MCTS_self_play(game_queue, connectnet, cpu, configs, iteration, device, progress_queue=None):
    """
    Perform self-play games using MCTS and the neural network, generating training data
    
    Args:
        game_queue: Queue of game indices to process
        connectnet: Neural network model
        cpu: CPU ID for this process
        configs: Configuration parameters
        iteration: Current training iteration
        device: Device to run computations on
        progress_queue: Queue to report progress for the progress bar
    """
    os.makedirs("datasets/iter_%d" % iteration, exist_ok=True)
    os.makedirs("games", exist_ok=True)
    
    while True:
        try:
            idxx = game_queue.get_nowait()
        except queue.Empty:
            break
            
        current_board = Board(num_cols=configs['board']['num_cols'], num_rows=configs['board']['num_rows'], win_streak=configs['board']['win_streak'])
        checkmate = False
        dataset = [] # to get state, policy, value for neural network training
        states = []
        value = 0
        move_count = 0
        
        # Create a game replay file for the first 5 games for debugging
        game_replay = []
        should_log = (cpu == 0 and idxx < 5)
        
        while checkmate == False and current_board.actions() != []:
            # high temperature for initial moves for exploration
            if move_count < configs['mcts']['initial_move_count']:
                t = configs['mcts']['temperature_MCTS']
            # low temperature for later moves for exploitation
            else:
                t = 0.1

            states.append(copy.deepcopy(current_board.current_board))
            board_state = copy.deepcopy(encode_board(current_board))
            root = UCT_search(current_board, configs['mcts']['num_simulations'], connectnet, t, device, c_puct=configs['self_play']['c_puct'], batch_size=configs['self_play']['batch_size'])
            policy = get_policy(root, t)

            if should_log:
                game_replay.append(f"Game {idxx} Move {move_count} POLICY:\n{policy}")

            current_board.drop_piece(np.random.choice(np.arange(current_board.num_cols), p = policy)) # decode move and move piece(s)
            dataset.append([board_state,policy])

            if should_log:
                game_replay.append(f"[Iteration: {iteration} CPU: {cpu}]: Game {idxx} Move {move_count} CURRENT BOARD:")
                game_replay.append(str(current_board.current_board))
                
            if current_board.check_winner() == True: # if somebody won
                if current_board.player == 0: # black wins
                    value = -1
                elif current_board.player == 1: # white wins
                    value = 1
                checkmate = True
            move_count += 1

        # Write game log to file
        if should_log:
            game_replay_path = f"./games/game_iter{iteration}_{idxx}_{configs['board']['num_cols']}_{configs['board']['num_rows']}_{configs['board']['win_streak']}.txt"
            with open(game_replay_path, 'w') as f:
                f.write('\n'.join(game_replay))

        dataset_p = [(s,p,value) for s,p in dataset]
        del dataset
        save_as_pickle("iter_%d/" % iteration + "dataset_iter%d_cpu%i_%i_%s" % (iteration, cpu, idxx, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p)
        
        # Update progress bar
        if progress_queue is not None:
            progress_queue.put(1)
   
def run_MCTS(configs, connectnet, start_idx=0, iteration=0, device="cuda"):
    """
    Run multiple MCTS self-play processes in parallel
    
    Args:
        configs: Configuration parameters
        connectnet: Neural network model
        start_idx: Starting index for game numbering
        iteration: Current training iteration
        device: Device to run computations on
    """
    connectnet.share_memory()
    mp.set_start_method('spawn', force=True)

    n_procs = configs['self_play']['MCTS_num_processes']
    total_games = configs['self_play']['num_games']
    
    # Create a queue of game indices
    game_queue = mp.Queue()
    for i in range(total_games):
        game_queue.put(start_idx + i)
    
    # Create a queue for progress tracking
    progress_queue = mp.Queue()
    
    # Create progress bar
    pbar = tqdm(total=total_games, desc="Self-play games")
    
    procs = []
    for cpu in range(n_procs):
        p = mp.Process(
            target=MCTS_self_play,
            args=(game_queue, connectnet, cpu, configs, iteration, device, progress_queue),
        )
        p.start()
        procs.append(p)
    
    # Update progress bar based on completed games
    completed_games = 0
    while completed_games < total_games:
        try:
            progress_queue.get(timeout=0.1)
            completed_games += 1
            pbar.update(1)
        except queue.Empty:
            # Check if all processes are still alive
            if not any(p.is_alive() for p in procs):
                break
    
    for p in procs:
        p.join()
    
    pbar.close()
