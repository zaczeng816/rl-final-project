#!/usr/bin/env python

import numpy as np
from connect_board import Board

def encode_board(board):
    board_state = board.current_board
    encoded = np.zeros([board.num_rows, board.num_cols, 3]).astype(int)
    encoder_dict = {"O":0, "X":1}
    for row in range(board.num_rows):
        for col in range(board.num_cols):
            if board_state[row,col] != " ":
                encoded[row, col, encoder_dict[board_state[row,col]]] = 1
    if board.player == 1:
        encoded[:,:,2] = 1 # player to move
    return encoded

def decode_board(encoded, num_rows=6, num_cols=7):
    decoded = np.zeros([num_rows, num_cols]).astype(str)
    decoded[decoded == "0.0"] = " "
    decoder_dict = {0:"O", 1:"X"}
    for row in range(num_rows):
        for col in range(num_cols):
            for k in range(2):
                if encoded[row, col, k] == 1:
                    decoded[row, col] = decoder_dict[k]
    cboard = Board()
    cboard.current_board = decoded
    cboard.player = encoded[0,0,2]
    return cboard