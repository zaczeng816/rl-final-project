#!/usr/bin/env python

import numpy as np

class Board():
    def __init__(self, num_rows=6, num_cols=7):
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.init_board = np.zeros([self.num_rows, self.num_cols]).astype(str)
        self.init_board[self.init_board == "0.0"] = " "

        self.player = 0
        self.current_board = self.init_board
        
    def num_rows(self):
        return self.num_rows
    
    def num_cols(self):
        return self.num_cols
    
    def drop_piece(self, column):
        if self.current_board[0, column] != " ":
            return "Invalid move"
        else:
            row = 0; pos = " "
            while (pos == " "):
                if row == self.num_rows:
                    row += 1
                    break
                pos = self.current_board[row, column]
                row += 1
            if self.player == 0:
                self.current_board[row-2, column] = "O"
                self.player = 1
            elif self.player == 1:
                self.current_board[row-2, column] = "X"
                self.player = 0
    
    def check_winner(self):
        if self.player == 1:
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    if self.current_board[row, col] != " ":
                        # rows
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col] == "O" and \
                                self.current_board[row + 2, col] == "O" and self.current_board[row + 3, col] == "O":
                                #print("row")
                                return True
                        except IndexError:
                            next
                        # columns
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row, col + 1] == "O" and \
                                self.current_board[row, col + 2] == "O" and self.current_board[row, col + 3] == "O":
                                #print("col")
                                return True
                        except IndexError:
                            next
                        # \ diagonal
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col + 1] == "O" and \
                                self.current_board[row + 2, col + 2] == "O" and self.current_board[row + 3, col + 3] == "O":
                                #print("\\")
                                return True
                        except IndexError:
                            next
                        # / diagonal
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col - 1] == "O" and \
                                self.current_board[row + 2, col - 2] == "O" and self.current_board[row + 3, col - 3] == "O"\
                                and (col-3) >= 0:
                                #print("/")
                                return True
                        except IndexError:
                            next

        if self.player == 0:
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    if self.current_board[row, col] != " ":
                        # rows
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col] == "X" and \
                                self.current_board[row + 2, col] == "X" and self.current_board[row + 3, col] == "X":
                                return True
                        except IndexError:
                            next
                        # columns
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row, col + 1] == "X" and \
                                self.current_board[row, col + 2] == "X" and self.current_board[row, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        # \ diagonal
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col + 1] == "X" and \
                                self.current_board[row + 2, col + 2] == "X" and self.current_board[row + 3, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        # / diagonal
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col - 1] == "X" and \
                                self.current_board[row + 2, col - 2] == "X" and self.current_board[row + 3, col - 3] == "X"\
                                and (col-3) >= 0:
                                return True
                        except IndexError:
                            next

    def actions(self): # returns all possible moves
        acts = []
        for col in range(self.num_cols):
            if self.current_board[0, col] == " ":
                acts.append(col)
        return acts
            