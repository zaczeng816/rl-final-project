import argparse
import yaml
import numpy as np
import importlib
from connect_four_env import ConnectFourEnv
from agents import HeuristicAgent, AlphaZeroAgent, RandomAgent



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 2 agents on the new custom environment")

    # Main player
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--model_checkpoint", "-mc",  required=True)

    # Opponent
    parser.add_argument("--oppo", "-o", required=True, help="Opponent agent: RandomAgent, HeuristicAgent")

    # Evaluation options
    parser.add_argument("--win_length", "-w", type=int, default=4, help="Number to connect in order to win ")
    parser.add_argument("--row", "-row", type=int, default=6, help="Board Row")
    parser.add_argument("--col", "-col", type=int, default=7, help="Board Col")
    parser.add_argument("--games", "-n", type=int, default=100, help="Number of games to play")
    parser.add_argument("--render", "-r",  default=True, help="Render ASCII board each move (if the env supports it)")

    return parser.parse_args()

def main():
    args = parse_args()
    
    opponent_agent = RandomAgent()
    if args.oppo == "HeuristicAgent":
        opponent_agent = HeuristicAgent()

    alpha_agent = AlphaZeroAgent(args.config, args.model_checkpoint)

    # Create environment
    env = ConnectFourEnv(
        render_mode="human" if args.render else None,
        opponent=opponent_agent,
        opponent_name=args.oppo,
        main_player_name="AlphaZeroAgent",
        first_player=1,
        main_player_id=1,
        win_length=args.win_length
    )

    wins = losses = draws = 0

    for game in range(args.games):
        print(f"\n=== Game {game+1}/{args.games} ===")
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            if env.current_player == env.main_player_id:
                action = alpha_agent.play(env)
            else:
                action = opponent_agent.play(env)

            obs, reward, done, *_ = env.step(action)
            ep_reward += reward

            # Optional ASCII render
            if args.render:
                env.render()

        # Report game result
        print(f"--- Game {game+1} finished: reward = {ep_reward} ---")
        if ep_reward > 0:
            wins += 1
            print(f"{env.main_player_name} won!")
        elif ep_reward < 0:
            losses += 1
            print(f"{env.opponent_name} won!")
        else:
            draws += 1
            print("Draw!")

    # Final summary
    print(f"\nOut of {args.games} games → wins: {wins}, losses: {losses}, draws: {draws}")

if __name__ == "__main__":
    main()