#!/usr/bin/env python
import argparse
import torch
import numpy as np
import copy
import yaml

from MCTS import UCT_search, get_policy
from model import ConnectNet
from connect_board import Board as cboard, encode_board


def play_game(net, configs, device):
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
        moves_count += 1
        dataset.append(copy.deepcopy(encode_board(current_board)))
        print(current_board.current_board); print(" ")
        if current_board.player == 0:
            if white != None:
                print("AI is thinking........")
                root = UCT_search(current_board,configs['mcts']['num_simulations'],white,t,device)
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
                root = UCT_search(current_board,configs['mcts']['num_simulations'],black,t,device)
                policy = get_policy(root, t)
            else:
                while(True):
                    col = input("Which column do you wanna drop your piece? (Enter 1-{} )\n".format(current_board.num_cols))
                    if int(col) in np.arange(current_board.num_cols) + 1:
                        policy = np.zeros([current_board.num_cols], dtype=np.float32); policy[int(col)-1] += 1
                        break
        current_board.drop_piece(np.random.choice(np.arange(current_board.num_cols), p = policy))
        if current_board.check_winner() == True: # someone wins
            if current_board.player == 0: # black wins
                value = -1
            elif current_board.player == 1: # white wins
                value = 1
            checkmate = True

    dataset.append(encode_board(current_board))
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="model_ckpts/c4_current_net_trained_iter8.pth.tar", help="Path to the trained network")
    parser.add_argument("--config", type=str, default="configs/h6_w7_c4_base.yaml", help="Path to the config file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    configs = yaml.safe_load(open(args.config, 'r'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ConnectNet(
        num_cols=configs['board']['num_cols'], 
        num_rows=configs['board']['num_rows'], 
        num_blocks=configs['model']['num_blocks']
    ).to(device)
    net.eval()

    checkpoint = torch.load(args.net)
    if 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(checkpoint)
    play_again = True
    while(play_again == True):
        play_game(net, configs, device)
        while(True):
            again = input("Do you wanna play again? (Y/N)\n")
            if again.lower() in ["y", "n"]:
                if again.lower() == "n":
                    play_again = False; break
                else:
                    break
                