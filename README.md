# rl-final-project

## Python Environment Setup

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

## Interactive Game Play On Web

#### Requirements
`npm` and `docker` are required to run the frontend and backend servers.

#### File Structure
`app.py` is the main file for the backend server using `fastapi`.
`frontend/` is the folder for the frontend server using `next.js`.

#### Running the servers
Run `start_backend.sh` to start the backend server.  
- this will start a fastapi server and a redis on docker which stores the game states

Run `start_frontend.sh` to start the frontend server.
- this will install frontend dependencies and start a Next.js app on `http://localhost:3000`

Open `http://localhost:3000` in your browser to play the game.

## Acknowledgements

The code structure and implementation is inspired by the following repository:
- [AlphaZero Connect4](https://github.com/plkmo/AlphaZero_Connect4) by plkmo - Used as reference for the AlphaZero algorithm implementation and neural network architecture