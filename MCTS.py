#!/usr/bin/env python
import pickle
import os
import collections
import numpy as np
import math
import copy
import torch
import torch.multiprocessing as mp
import datetime
import logging

from tqdm import tqdm

import encoder_decoder_c4 as ed
from connect_board import Board as c_board

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def save_as_pickle(filename, data):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(filename):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class UCTNode():
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
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value
    
    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value
    
    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self):
        return math.sqrt(self.number_visits) * (
            abs(self.child_priors) / (1 + self.child_number_visits))
    
    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove
    
    def select_leaf(self):
        current = self
        while current.is_expanded:
          best_move = current.best_child()
          current = current.maybe_add_child(best_move)
        return current
    
    def add_dirichlet_noise(self,action_idxs,child_priors):
        valid_child_priors = child_priors[action_idxs] # select only legal moves entries in child_priors array
        valid_child_priors = 0.75*valid_child_priors + 0.25*np.random.dirichlet(np.zeros([len(valid_child_priors)], \
                                                                                          dtype=np.float32)+192)
        child_priors[action_idxs] = valid_child_priors
        return child_priors
    
    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = self.game.actions(); c_p = child_priors
        if action_idxs == []:
            self.is_expanded = False
        self.action_idxes = action_idxs
        c_p[[i for i in range(len(child_priors)) if i not in action_idxs]] = 0.000000000 # mask all illegal actions
        if self.parent.parent == None: # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_idxs,c_p)
        self.child_priors = c_p
    
    def decode_n_move_pieces(self,board,move):
        board.drop_piece(move)
        return board
            
    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game) # make copy of board
            copy_board = self.decode_n_move_pieces(copy_board,move)
            self.children[move] = UCTNode(
              copy_board, move, parent=self)
        return self.children[move]
    
    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1: # same as current.parent.game.player = 0
                current.total_value += (1*value_estimate) # value estimate +1 = O wins
            elif current.game.player == 0: # same as current.parent.game.player = 1
                current.total_value += (-1*value_estimate)
            current = current.parent
        
class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)

class BatchedEvaluator:
    def __init__(self, net, device, batch_size=64):
        self.net = net
        self.device = device
        self.batch_size = batch_size
        self._buf_states = []
        self._buf_leaves = []

    def enqueue(self, leaf):
        s = ed.encode_board(leaf.game)
        t = torch.from_numpy(s.transpose(2,0,1)).float()
        self._buf_states.append(t)
        self._buf_leaves.append(leaf)
        if len(self._buf_states) >= self.batch_size:
            self.flush()

    @torch.no_grad()
    def flush(self):
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


def UCT_search(game_state, num_reads, net, temp, device):
    root = UCTNode(game_state, move=None, parent=DummyNode())
    evaluator = BatchedEvaluator(net, device)
    for _ in range(num_reads):
        leaf = root.select_leaf()
        evaluator.enqueue(leaf)
    evaluator.flush()
    return root

def do_decode_n_move_pieces(board,move):
    board.drop_piece(move)
    return board

def get_policy(root, temp=1):
    return ((root.child_number_visits)**(1/temp))/sum(root.child_number_visits**(1/temp))

@torch.no_grad()
def MCTS_self_play(connectnet, num_games, start_idx, cpu, configs, iteration, device):
    logger.info("[CPU: %d]: Starting MCTS self-play..." % cpu)
    
    if not os.path.isdir("./datasets/iter_%d" % iteration):
        if not os.path.isdir("datasets"):
            os.mkdir("datasets")
        os.mkdir("datasets/iter_%d" % iteration)
        
    for idxx in tqdm(range(start_idx, num_games + start_idx), position=cpu):
        # logger.info("[CPU: %d]: Game %d" % (cpu, idxx))
        current_board = c_board(num_cols=configs['board']['num_cols'], num_rows=configs['board']['num_rows'], win_streak=configs['board']['win_streak'])
        checkmate = False
        dataset = [] # to get state, policy, value for neural network training
        states = []
        value = 0
        move_count = 0
        while checkmate == False and current_board.actions() != []:
            if move_count < 11:
                t = configs['mcts']['temperature_MCTS']
            else:
                t = 0.1
            states.append(copy.deepcopy(current_board.current_board))
            board_state = copy.deepcopy(ed.encode_board(current_board))
            root = UCT_search(current_board, configs['mcts']['num_simulations'], connectnet, t, device)
            policy = get_policy(root, t)
            current_board = do_decode_n_move_pieces(current_board,\
                                                    np.random.choice(np.arange(current_board.num_cols), \
                                                                     p = policy)) # decode move and move piece(s)
            dataset.append([board_state,policy])
            if current_board.check_winner() == True: # if somebody won
                if current_board.player == 0: # black wins
                    value = -1
                elif current_board.player == 1: # white wins
                    value = 1
                checkmate = True
            move_count += 1
        dataset_p = []
        for idx,data in enumerate(dataset):
            s,p = data
            if idx == 0:
                dataset_p.append([s,p,0])
            else:
                dataset_p.append([s,p,value])
        del dataset
        save_as_pickle("iter_%d/" % iteration +\
                       "dataset_iter%d_cpu%i_%i_%s" % (iteration, cpu, idxx, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p)
   
def run_MCTS(configs, connectnet, start_idx=0, iteration=0, device="cuda"):
    connectnet.share_memory()
    mp.set_start_method('spawn', force=True)

    n_procs = configs['self_play']['MCTS_num_processes']
    games_per_proc = configs['self_play']['num_games_per_MCTS_process']
    procs = []
    for cpu in range(n_procs):
        s = start_idx + cpu * games_per_proc
        p = mp.Process(
            target=MCTS_self_play,
            args=(connectnet, games_per_proc, s, cpu, configs, iteration, device),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
