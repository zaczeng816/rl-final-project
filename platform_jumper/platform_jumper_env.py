import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class PlatformJumperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # Define the action space (0: wait, 1: left, 2: right, 3: jump)
        self.action_space = spaces.Discrete(4)

        # Observation: [player_x, player_y, player_velocity_y, next_platform_dx, next_platform_dy, next_platform_width]
        low_obs = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0], dtype=np.float32)
        high_obs = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)

        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)

        self.gravity = -0.5
        self.jump_velocity = 10
        self.player = {"x": 0, "y": 0, "velocity_y": 0}
        self.platforms = []
        self.score = 0

        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((400, 600))
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player = {"x": 0, "y": 0, "velocity_y": 0}
        self.platforms = [{"x": 0, "y": -10, "width": 10}]
        self.score = 0

        observation = self._get_obs()
        return observation, {}

    def _get_obs(self):
        next_plat = self.platforms[0]
        dx = next_plat["x"] - self.player["x"]
        dy = next_plat["y"] - self.player["y"]
        return np.array([self.player["x"], self.player["y"], self.player["velocity_y"],
                         dx, dy, next_plat["width"]], dtype=np.float32)

    def step(self, action):
        if action == 1:
            self.player["x"] -= 1
        elif action == 2:
            self.player["x"] += 1
        elif action == 3 and self._on_platform():
            self.player["velocity_y"] = self.jump_velocity

        # Gravity and velocity update
        self.player["velocity_y"] += self.gravity
        self.player["y"] += self.player["velocity_y"]

        reward = 0
        terminated = False

        if self.player["y"] < -50:  # fell off
            terminated = True
            reward = -10

        # Check if reached next platform
        if self._on_platform() and self.player["velocity_y"] <= 0:
            self.player["velocity_y"] = 0
            reward = 10
            self.score += 1
            self._generate_new_platform()

        observation = self._get_obs()
        return observation, reward, terminated, False, {}

    def _on_platform(self):
        for plat in self.platforms:
            if (abs(self.player["x"] - plat["x"]) <= plat["width"]/2 and
                    abs(self.player["y"] - plat["y"]) <= 1):
                return True
        return False

    def _generate_new_platform(self):
        new_x = np.random.uniform(-5, 5)
        new_y = self.player["y"] + np.random.uniform(10, 15)
        width = np.random.uniform(5, 10)
        self.platforms = [{"x": new_x, "y": new_y, "width": width}]

    def render(self):
        if self.render_mode == "human":
            self.window.fill((255, 255, 255))
            # Draw player
            pygame.draw.rect(self.window, (255, 0, 0), pygame.Rect(200 + self.player["x"]*10, 500 - self.player["y"]*10, 20, 20))
            # Draw platforms
            for plat in self.platforms:
                pygame.draw.rect(self.window, (0, 0, 0),
                                 pygame.Rect(200 + (plat["x"] - plat["width"]/2)*10, 500 - plat["y"]*10, plat["width"]*10, 5))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
