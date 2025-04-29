import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time

class ConnectFourEnv(gym.Env):
    """
    Gymnasium environment for Connect Four.
    Board values: 0 = empty, 1 = Player 1, 2 = Player 2.
    Reward: +1 for Player 1 win, -1 for Player 1 loss, 0 for draw/ongoing.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        opponent=None,
        render_mode=None,
        first_player=None,
        main_player_id=1,
        main_player_name="Main",
        opponent_name="Opponent",
        rows=6,
        cols=7,
        win_length=4
    ):
        super().__init__()

        # Board shape and rule
        self.ROWS = rows
        self.COLS = cols
        self.WIN_REWARD = 1
        self.WIN_LENGTH = win_length # how many in a row to win

        # check if render_mode is valid
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert main_player_id in [1, 2]
        assert first_player is None or first_player in [1, 2]

        self.opponent = opponent
        self.auto_opponent = opponent is not None

        # Which side our RL agent is (1 or 2)
        self.main_player_id = main_player_id
        self.opponent_id = 3 - main_player_id

        # If None → randomize who goes first; else fix to 1 or 2
        self.first_player = first_player

        # Rendering
        self.render_mode = render_mode
        self.main_player_name = main_player_name
        self.opponent_name = opponent_name
        self.window = None
        self.last_render_time = None

        # Gymnasium spaces
        self.action_space = spaces.Discrete(self.COLS)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.ROWS, self.COLS), dtype=np.int8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # empty board
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.last_move = None
        self.invalid_move = False

        # Decide who starts
        if self.first_player:
            self.current_player = self.first_player
        else:
            self.current_player = self.np_random.choice([1, 2])

        # If opponent goes first, let them play
        if self.auto_opponent and self.current_player == self.opponent_id:
            col = self.opponent.play(self.board.copy())
            self.place_piece(col)
            self.switch_player()

        if self.render_mode == "human":
            self.render()

        return self.board.copy(), {}

    def step(self, action):
        # Only called when it's the main player's turn
        assert self.current_player == self.main_player_id, "Not your turn"

        # Main agent move
        self.place_piece(action)
        self.switch_player()
        done, winner = self.check_board()
        if done:
            return self.board.copy(), self.reward(winner), done, False, {}

        # Opponent move
        if self.auto_opponent:
            col = self.opponent.play(self.board.copy())
            self.place_piece(col)
            self.switch_player()
            done, winner = self.check_board()
            if done:
                return self.board.copy(), self.reward(winner), done, False, {}

        # neither side has finished
        if self.render_mode == "human":
            self.render()
        return self.board.copy(), 0, False, False, {}

    def place_piece(self, action):
        if action < 0 or action >= self.COLS or self.board[0, action] != 0:
            self.invalid_move = True
            return

        for r in range(self.ROWS - 1, -1, -1):
            if self.board[r, action] == 0:
                self.board[r, action] = self.current_player
                self.last_move = (r, action)
                return

    def switch_player(self):
        self.current_player = 3 - self.current_player

    # checking that the game is still in progress
    def check_board(self):
        # invalid move → other player wins immediately
        if self.invalid_move:
            return True, 3 - self.current_player

        if self.last_move is None:
            return False, None

        r, c = self.last_move
        p = self.board[r, c]

        # check all directions on the board
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for dr, dc in directions:
            count = 0
            for offset in range(-(self.WIN_LENGTH-1), self.WIN_LENGTH):
                rr, cc = r + dr*offset, c + dc*offset
                if 0 <= rr < self.ROWS and 0 <= cc < self.COLS and self.board[rr, cc] == p:
                    count += 1
                    if count >= self.WIN_LENGTH:
                        return True, p
                else:
                    count = 0

        # draw
        if np.all(self.board != 0):
            return True, 0

        return False, None

    def reward(self, winner):
        if winner == self.main_player_id:
            return +self.WIN_REWARD
        elif winner == 0:
            return 0
        else:
            return -self.WIN_REWARD

    def render(self):

        # to be used to display in notebook/log
        if self.render_mode != "human":
            print("Connect Four board:")
            for row in self.board:
                # ‘⚪’ empty, ‘🟡' player 1, ‘🟣’ player 2
                symbols = {
                  0: '⚪',
                  1: '🟡',
                  2: '🟣'
                }
                print("| " + " ".join(symbols[c] for c in row) + " |")
            # Legend
            print("\nLegend:")
            print(f"  🟡 {self.main_player_name} (Player {self.main_player_id})")
            print(f"  🟣 {self.opponent_name} (Player {self.opponent_id})")
            return

        # initialize
        if self.window is None:
            pygame.init()
            size = 32
            pad = 4
            width = self.COLS * (2*size + pad) + pad
            height = self.ROWS * (2*size + pad) + pad + 80
            self.window = pygame.display.set_mode((width, height))

        # cap FPS
        now = time.time()
        target = 1 / self.metadata.get("render_fps", 4)
        if hasattr(self, "last_render_time") and self.last_render_time:
            delta = now - self.last_render_time
            if delta < target:
                time.sleep(target - delta)
        self.last_render_time = time.time()

        # Draw board background and circles
        canvas = pygame.Surface(self.window.get_size())
        canvas.fill((30, 144, 255))
        size = 32
        pad = 4
        for r in range(self.ROWS):
            for c in range(self.COLS):
                x = pad + c*(2*size+pad) + size
                y = pad + r*(2*size+pad) + size
                cell = self.board[r, c]
                if cell == 1:
                    color = (224, 209, 18)   # Player 1
                elif cell == 2:
                    color = (197, 7, 17)     # Player 2
                else:
                    color = (245, 245, 245)  # empty
                pygame.draw.circle(canvas, color, (x, y), size)

        font = pygame.font.Font(None, 36)
        legend_y = self.ROWS*(2*size+pad) + pad + 10
        # main player
        pygame.draw.circle(canvas, (224,209,18), (pad + size, legend_y + size//2), size//2)
        txt1 = font.render(f"{self.main_player_name}", True, (0,0,0))
        canvas.blit(txt1, (pad + 2*size, legend_y))
        # opponent
        opp_x = pad + 2*size + txt1.get_width() + 40
        pygame.draw.circle(canvas, (197,7,17), (opp_x, legend_y + size//2), size//2)
        txt2 = font.render(f"{self.opponent_name}", True, (0,0,0))
        canvas.blit(txt2, (opp_x + size, legend_y))

        self.window.blit(canvas, (0,0))
        pygame.display.flip()
        pygame.event.pump()

    def clone(self):
        new = ConnectFourEnv(
            opponent=self.opponent,
            render_mode=None,
            first_player=self.first_player,
            main_player_id=self.main_player_id,
            main_player_name=self.main_player_name,
            opponent_name=self.opponent_name,
        )
        new.board = self.board.copy()
        new.current_player = self.current_player
        new.last_move = self.last_move
        new.invalid_move = self.invalid_move
        return new

    def close(self):
        if self.window:
            pygame.quit()
