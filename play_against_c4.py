#!/usr/bin/env python

import os.path
import torch
import numpy as np
from alpha_net_c4 import ConnectNet
from connect_board import Board as cboard
import encoder_decoder_c4 as ed
import copy
from MCTS_c4 import UCT_search, do_decode_n_move_pieces, get_policy
import pickle
import datetime
import yaml

def play_game(net, configs):
    # Asks human what he/she wanna play as
    white = None; black = None
    while (True):
        play_as = input("What do you wanna play as? (\"O\"/\"X\")? Note: \"O\" starts first, \"X\" starts second\n")
        if play_as == "O":
            black = net; break
        elif play_as == "X":
            white = net; break
        else:
            print("I didn't get that.")
    current_board = cboard(num_cols=configs['board']['num_cols'], num_rows=configs['board']['num_rows'], win_streak=configs['board']['win_streak'])
    checkmate = False
    dataset = []
    value = 0; t = 0.1; moves_count = 0
    while checkmate == False and current_board.actions() != []:
        if moves_count <= 5:
            t = 1
        else:
            t = 0.1
        moves_count += 1
        dataset.append(copy.deepcopy(ed.encode_board(current_board)))
        print(current_board.current_board); print(" ")
        if current_board.player == 0:
            if white != None:
                print("AI is thinking........")
                root = UCT_search(current_board,configs['mcts']['num_simulations'],white,t)
                policy = get_policy(root, t)
            else:
                while(True):
                    col = input("Which column do you wanna drop your piece? (Enter 1-{})\n".format(current_board.num_cols))
                    if int(col) in np.arange(current_board.num_cols) + 1:
                        policy = np.zeros([current_board.num_cols], dtype=np.float32); policy[int(col)-1] += 1
                        break
        elif current_board.player == 1:
            if black != None:
                print("AI is thinking.............")
                root = UCT_search(current_board,configs['mcts']['num_simulations'],black,t)
                policy = get_policy(root, t)
            else:
                while(True):
                    col = input("Which column do you wanna drop your piece? (Enter 1-{} )\n".format(current_board.num_cols))
                    if int(col) in np.arange(current_board.num_cols) + 1:
                        policy = np.zeros([current_board.num_cols], dtype=np.float32); policy[int(col)-1] += 1
                        break
        current_board = do_decode_n_move_pieces(current_board,\
                                                np.random.choice(np.arange(current_board.num_cols), \
                                                                 p = policy)) # decode move and move piece(s)
        if current_board.check_winner() == True: # someone wins
            if current_board.player == 0: # black wins
                value = -1
            elif current_board.player == 1: # white wins
                value = 1
            checkmate = True
    dataset.append(ed.encode_board(current_board))
    print(current_board.current_board); print(" ")
    if value == -1:
        if play_as == "O":
            dataset.append(f"AI as black wins"); print("YOU LOSE!!!!!!!")
        else:
            dataset.append(f"Human as black wins"); print("YOU WIN!!!!!!!")
        return "black", dataset
    elif value == 1:
        if play_as == "O":
            dataset.append(f"Human as white wins"); print("YOU WIN!!!!!!!!!!!")
        else:
            dataset.append(f"AI as white wins"); print("YOU LOSE!!!!!!!")
        return "white", dataset
    else:
        dataset.append("Nobody wins"); print("DRAW!!!!!")
        return None, dataset

if __name__ == "__main__":
    best_net_filename = "model_data/cc3_current_net_iter26.pth.tar"
    configs = yaml.safe_load(open('configs/h5_w5_c3_small.yaml', 'r'))
    best_cnet = ConnectNet(num_cols=configs['board']['num_cols'], num_rows=configs['board']['num_rows'], num_blocks=configs['model']['num_blocks'])
    cuda = torch.cuda.is_available()
    if cuda:
        best_cnet.cuda()
    best_cnet.eval()
    checkpoint = torch.load(best_net_filename)
    best_cnet.load_state_dict(checkpoint['state_dict'])
    play_again = True
    while(play_again == True):
        play_game(best_cnet, configs)
        while(True):
            again = input("Do you wanna play again? (Y/N)\n")
            if again.lower() in ["y", "n"]:
                if again.lower() == "n":
                    play_again = False; break
                else:
                    break
                