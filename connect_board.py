#!/usr/bin/env python
import numpy as np

class Board():
    ''' Connect 4 board used in self-play'''

    def __init__(self, num_rows=6, num_cols=7, win_streak=4):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.win_streak = win_streak

        self.init_board = np.zeros([self.num_rows, self.num_cols]).astype(str)
        self.init_board[self.init_board == "0.0"] = " "

        self.player = 0
        self.current_board = self.init_board
        
    def num_rows(self):
        return self.num_rows
    
    def num_cols(self):
        return self.num_cols
    
    def drop_piece(self, col: int) -> bool:
        ''' Drop a piece in the column
        Args:
            col: int - column to drop the piece
        Returns:
            bool - True if the move is valid, False otherwise
        '''
        # illegal move
        if not (0 <= col < self.num_cols) or self.current_board[0, col] != " ":
            raise ValueError("Invalid move")

        # drop piece
        for r in range(self.num_rows - 1, -1, -1):
            if self.current_board[r, col] == " ":
                self.current_board[r, col] = "O" if self.player == 0 else "X"
                self.player ^= 1
                return True
        raise ValueError("Invalid move")
    
    def check_winner(self):
        ''' Check if a player has won the game on the current board
        Returns:
            bool - True if a player has won, False otherwise
        '''
        target = "O" if self.player == 1 else "X"  # Check for the previous player's pieces

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if self.current_board[r, c] != target:
                    continue

                # Try each direction
                for dr, dc in directions:
                    count = 1
                    for i in range(1, self.win_streak):
                        rr, cc = r + dr * i, c + dc * i
                        if (
                            0 <= rr < self.num_rows and
                            0 <= cc < self.num_cols and
                            self.current_board[rr, cc] == target
                        ):
                            count += 1
                        else:
                            break

                    if count >= self.win_streak:
                        return True

        return False

    def actions(self):
        ''' Returns the list of columns that are playable e.g. not full'''
        acts = []
        for col in range(self.num_cols):
            if self.current_board[0, col] == " ":
                acts.append(col)
        return acts

# static methods
def encode_board(board):
    ''' Encode the board into representation used by the neural network
    Args:
        board: Board - the board to encode
    Returns:
        np.ndarray - the encoded board
    '''
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
    ''' Decode the board from the representation used by the neural network
    Args:
        encoded: np.ndarray - the encoded board
        num_rows: int - the number of rows in the board
        num_cols: int - the number of columns in the board
    Returns:
        Board - the decoded board
    '''
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