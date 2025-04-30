import argparse
import yaml
import numpy as np
from connect_four_env import ConnectFourEnv
from agents import HeuristicAgent, AlphaZeroAgent, RandomAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 2 agents on the new custom environment")

    # Main player
    parser.add_argument("--agent", "-a", choices=['RandomAgent', 'HeuristicAgent', 'AlphaZeroAgent'], required=True)
    parser.add_argument("--config", "-c", required=False)
    parser.add_argument("--model_checkpoint", "-mc",  required=False)

    # Opponent
    parser.add_argument("--opponent", "-o", choices=['RandomAgent', 'HeuristicAgent', 'AlphaZeroAgent'], help="Opponent agent: RandomAgent, HeuristicAgent")

    # Evaluation options
    parser.add_argument("--win_length", "-w", type=int, default=4, help="Number to connect in order to win ")
    parser.add_argument("--row", "-row", type=int, default=6, help="Board Row")
    parser.add_argument("--col", "-col", type=int, default=7, help="Board Col")
    parser.add_argument("--games", "-n", type=int, default=100, help="Number of games to play")
    parser.add_argument("--render", "-r",  default=False, help="Render ASCII board each move (if the env supports it)")
    parser.add_argument("--verbose", "-v",  type=int, default=0, help="Verbose level per game (0-7)")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.agent == "HeuristicAgent":
        agent = HeuristicAgent()
    elif args.agent == "RandomAgent":
        agent = RandomAgent()
    elif args.agent == "AlphaZeroAgent":
        if args.config is None or args.model_checkpoint is None:
            raise ValueError("Please provide a config and model checkpoint for AlphaZeroAgent")
        agent = AlphaZeroAgent(args.config, args.model_checkpoint)


    if args.opponent == "HeuristicAgent":
        opponent_agent = HeuristicAgent()
    elif args.opponent == "RandomAgent":
        opponent_agent = RandomAgent()
    elif args.agent == "AlphaZeroAgent":
        if args.config is None or args.model_checkpoint is None:
            raise ValueError("Please provide a config and model checkpoint for AlphaZeroAgent")
        opponent_agent = AlphaZeroAgent(args.config, args.model_checkpoint)

    # Create environment
    env = ConnectFourEnv(
        render_mode="human" if args.render else None,
        opponent=opponent_agent,
        opponent_name=args.opponent,
        main_player_name="AlphaZeroAgent",
        main_player_id=1,
        win_length=args.win_length
    )

    wins = losses = draws = 0

    for game in range(args.games):
        if args.verbose > 0: print(f"\n=== Game {game+1}/{args.games} ===")
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            if env.current_player == env.main_player_id:
                action = agent.play(env)
            else:
                action = opponent_agent.play(env)

            obs, reward, done, *_ = env.step(action)
            ep_reward += reward

            # Optional ASCII render
            if args.render:
                env.render()

        # Report game result
        if args.verbose > 0:
            print(f"Game {game+1} ended with reward: {ep_reward}")        
        
        if ep_reward > 0:
            wins += 1
            if args.verbose > 0: print(f"{env.main_player_name} won!")
        elif ep_reward < 0:
            losses += 1
            if args.verbose > 0: print(f"{env.opponent_name} won!")
        else:
            draws += 1
            if args.verbose > 0: print("Draw!")

    # Final summary
    print("\n=== Final Results ===")
    print('Agent:', args.agent)
    print('Opponent:', args.opponent)
    print('Board:', args.row, 'x', args.col)
    print(f"Out of {args.games} games → wins: {wins}, losses: {losses}, draws: {draws}, winrate: {wins/args.games:.2%}")

if __name__ == "__main__":
    main()