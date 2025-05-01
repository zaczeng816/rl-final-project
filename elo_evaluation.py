"""
Elo evaluation of various agents on a customized Connect Four Gym environment.
Saves a fixed starting Elo for each agent, plays head-to-head matches, updates ratings, and prints a final leaderboard.

"""

import argparse
from connect_four_env import ConnectFourEnv
from agents import AlphaZeroAgent, HeuristicAgent, RandomAgent, ChildPlayer, BabyPlayer, ChildSmarterPlayer, BabySmarterPlayer

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace:  
            - config           Path to AlphaZeroAgent YAML config file.  
            - model_checkpoint Path to AlphaZeroAgent model checkpoint.  
            - win_length       Number in a row required to win.  
            - row              Number of rows on the board.  
            - col              Number of columns on the board.  
            - games            Number of games per agent pairing.  
            - k_factor         Elo K-factor for rating updates.
    """
    parser = argparse.ArgumentParser(description="Evaluate agents with Elo rating on the new custom environment")

    # alphazeroagent
    parser.add_argument("--config", "-c", help="AlphaZeroAgent Config", required=True)
    parser.add_argument("--model_checkpoint", "-mc", help="AlphaZeroAgent Checkpoint", required=True)

    # Evaluation options
    parser.add_argument("--win_length", "-w", type=int, default=4, help="Number to connect in order to win ")
    parser.add_argument("--row", "-row", type=int, default=6, help="Board Row")
    parser.add_argument("--col", "-col", type=int, default=7, help="Board Col")
    parser.add_argument("--games", "-n", type=int, default=100, help="Number of games to play")
    parser.add_argument("--k_factor", "-kf", type=int, default=32, help="k factor for updating elo")

    return parser.parse_args()


def update_elo(ra, rb, score_a, k):

    """
    Perform a two-player Elo rating update.

    Args:
        ra (float): Player 1’s current rating.
        rb (float): Player 2’s current rating.
        score_a (float): Actual score for player 1.
        k (float): K-factor controlling update magnitude.

    Returns:
        Tuple[float, float]: (ra_new, rb_new) updated ratings for players 1 and 2.
    """

    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    ra_new = ra + k * (score_a - ea)
    rb_new = rb + k * ((1 - score_a) - eb)
    return ra_new, rb_new

def play_match(agent1, agent2, num_games, env_kwargs):

    """
    Play a head-to-head match of `num_games` between two agents.

    Args:
        agent1 (Player): Agent assigned to play as Player 1.
        agent2 (Player): Agent assigned to play as Player 2.
        num_games (int): Number of games in the match.
        env_kwargs (dict): Arguments to pass into ConnectFourEnv constructor.

    Returns:
        Tuple[int, int, int]: (wins1, wins2, draws) tally of results.
    """

    env = ConnectFourEnv(**env_kwargs)

    wins1 = wins2 = draws = 0

    for game in range(num_games):
        print(f"\n=== Game {game+1}/{num_games} ===")
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            if env.current_player == env.main_player_id:
                action = agent1.play(env)
            else:
                action = agent2.play(env)

            obs, reward, done, *_ = env.step(action)
            ep_reward += reward

        # Report game result
        print(f"Game {game+1} ended with reward: {ep_reward}")        
        
        if ep_reward > 0:
            wins1 += 1
            print(f"{env.main_player_name} won!")
        elif ep_reward < 0:
            wins2 += 1
            print(f"{env.opponent_name} won!")
        else:
            draws += 1
            print("Draw!")


    return wins1, wins2, draws


def main():
    
    args = parse_args()

    AGENT_CLASSES = [
        
        HeuristicAgent,
        RandomAgent,
        AlphaZeroAgent,
        BabyPlayer,
        ChildPlayer,
        BabySmarterPlayer,
        ChildSmarterPlayer
    ]

    agents_list = []

    for cls in AGENT_CLASSES:
        if cls.__name__.startswith("AlphaZeroAgent"):
            inst = cls(args.config, args.model_checkpoint)
        else:
            inst = cls()
        agents_list.append((cls.__name__, inst))
    
    elos = {name: 1200.0 for name, _ in agents_list}

    for i in range(len(agents_list)):
        name_i, agent_i = agents_list[i]
        for j in range(i+1, len(agents_list)):
            name_j, agent_j = agents_list[j]

            env_kwargs = dict(
              opponent       = agent_j,
              main_player_id = 1,
              main_player_name = name_i,
              opponent_name = name_j,
              first_player   = None,
              win_length     = args.win_length
            )

            print(f"{name_i} vs {name_j}:")

            w_i, w_j, d = play_match(agent_i, agent_j, args.games, env_kwargs)
            score_i = (w_i + 0.5*d) / args.games
            new_i, new_j = update_elo(
                elos[name_i], elos[name_j], score_i, args.k_factor
            )
            elos[name_i], elos[name_j] = new_i, new_j
            print(f"{name_i} wins: {w_i} {name_j} wins: {w_j} draws: {d}  →  Elo: {name_i} {new_i:.1f}, {name_j} {new_j:.1f}")
            print(f"============================================================================")
            print(f"============================================================================")

    # Final Elo ladder board
    print("\nFinal Elo ladder:")
    for name, _ in sorted(elos.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:20s} {elos[name]:.1f}")

if __name__ == "__main__":
    main()