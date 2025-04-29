
from connect_four_env import ConnectFourEnv
from agents import RandomAgent, SmartAgent
import numpy as np


    
def play_game(agent, opponent):
    env = ConnectFourEnv(opponent=opponent)
    env.reset()
    
    done = False
    while not done:
        action = agent.play(env)
        obs, reward, done, _, _ = env.step(action)
    
    return reward

def evaluate_agent(agent, opponent, num_games=1000):
    total_reward = 0
    for _ in range(num_games):
        reward = play_game(agent, opponent)
        total_reward += reward
    return total_reward / num_games


num_games = 1000
agent = RandomAgent()
opponent = RandomAgent()
reward = evaluate_agent(agent, opponent, num_games=num_games)
print(f"Average reward of RandomAgent over {num_games} games: {reward}")



agent = SmartAgent()
opponent = RandomAgent()
reward = evaluate_agent(agent, opponent, num_games=num_games)
print(f"Average reward of SmartAgent over {num_games} games: {reward}")