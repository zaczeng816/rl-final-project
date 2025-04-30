#!/usr/bin/env python

import numpy as np

class Board():
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
        target = "O" if self.player == 1 else "X"

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

    def actions(self): # returns all possible moves
        acts = []
        for col in range(self.num_cols):
            if self.current_board[0, col] == " ":
                acts.append(col)
        return acts
            