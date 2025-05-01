# rl-final-project

## Setup

```bash
conda create -n rl-final python==3.10 -y
```

```bash
conda activate rl-final
```

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --config configs/h6_w7_c4_small_200.yaml --device cuda
```

## Evaluation

### Evaluate 2 agents:
Baseline agents:
- RandomAgent: selects uniform random valid moves.
- HeuristicAgent: rule-based heuristic for win/block/threats.
```bash
python evaluate_2agents.py --agent 'RandomAgent' --opponent 'HeuristicAgent'
```

AlphaZeroAgent vs. selected agent:
```bash
python evaluate_2agents.py --config configs/h6_w7_c4_small_200.yaml --model_checkpoint final_ckpt/h6_w7_c4_current_net_small_200_step80000.pth --opponent <pick an agent>
```

### Elo ladder:
```bash
python elo_evaluation.py --config configs/h6_w7_c4_small_200.yaml --model_checkpoint final_ckpt/h6_w7_c4_current_net_small_200_step80000.pth
```

## Agents

### AlphaZeroAgent
#### Overview
AlphaZeroAgent is an advanced Connect4-playing agent built on deep reinforcement learning and Monte Carlo Tree Search (MCTS), inspired by the original AlphaZero algorithm from DeepMind. It leverages a powerful deep neural network to guide its search for optimal moves and is capable of learning from self-play without any handcrafted rules or prior human knowledge.

This agent combines planning (via MCTS) and generalization (via a neural network) to play Connect4 at a high level.

#### 🚀 How It Works

The AlphaZeroAgent works by simulating thousands of potential future game states from the current board using Monte Carlo Tree Search. Each simulated state is evaluated by a deep residual convolutional neural network, which outputs:

A policy vector (probability distribution over legal moves)

A value (predicted outcome of the game from the current player's perspective)

The neural network guides the tree search and improves efficiency by focusing on promising moves. The agent then samples a move based on the visit counts of the search tree, with added randomness controlled by a temperature parameter to balance exploration and exploitation.

### Heuristic agent
The agent's decision-making process follows a prioritized heuristic approach to select the most optimal move available at any point during the game. Here's a breakdown of the strategy:

#### Win Immediately
If there's a move that allows the agent to win instantly, it takes it without hesitation.

#### Block Immediate Threats
If the opponent has a potential winning move on their next turn, the agent blocks it.

#### Avoid Dangerous Moves
The agent checks if its move would allow the opponent to win immediately afterward (e.g., by stacking on top of it). Such risky moves are avoided when possible.

#### Create Future Opportunities
The agent seeks to build strong positions by placing tokens that form a chain of (win_length - 1) with open ends, increasing the chances of winning in future turns.

#### Fallback Strategy
If no immediate win, block, or setup is found, the agent selects a valid random move, preferring safer positions. If all moves are risky, it chooses the least harmful option.

### Child Player Agent
ChildPlayer is a simple heuristic-based Connect4 agent designed to demonstrate basic strategic thinking. It follows a two-step decision process:

- Immediate Win Check: It scans all possible valid moves and plays the first one that results in an instant win.

- Fallback to Random: If no winning move is found, it selects randomly from the set of legal moves.


### Baby Player Agent
BabyPlayer is the most basic Connect4 agent, relying entirely on randomness. It chooses any column at random, without considering the current state of the board, possible wins, losses, or legal move validity (assumed to be handled by the environment).

- No Heuristics: Makes no attempt to win or block.

- No Game Awareness: Does not analyze the board or opponent's strategy.

- Non-Deterministic: Every move is randomly chosen, leading to unpredictable behavior.

---

## Interactive Game Play On Web

#### Requirements
`npm` and `docker` are required to run the frontend and backend servers.

#### File Structure
`app.py` is the main file for the backend server using `fastapi`.
`frontend/` is the folder for the frontend server using `next.js`.

#### Running the servers
Use `start_backend.sh` to start the backend server.  
- this will start a fastapi server and a redis on docker which stores the game states

Use `start_frontend.sh` to start the frontend server.
- this will install frontend dependencies and start a react app on `http://localhost:3000`

Open `http://localhost:3000` in your browser to play the game.

## Acknowledgements

The code structure and implementation is inspired by the following repository:
- [AlphaZero Connect4](https://github.com/plkmo/AlphaZero_Connect4) by plkmo - Used as reference for the AlphaZero algorithm implementation and neural network architecture