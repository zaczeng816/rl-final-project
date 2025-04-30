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
python train.py --config configs/h6_w7_c4_small.yaml --device cuda
```

## Playing against the trained model

```bash
python play_against_c4.py --net model_ckpts/cc4_current_net__iter7.pth.tar --config configs/h6_w7_c4_small.yaml
```

## Acknowledgements

The code structure and implementation is inspired by the following repository:
- [AlphaZero Connect4](https://github.com/plkmo/AlphaZero_Connect4) by plkmo - Used as reference for the AlphaZero algorithm implementation and neural network architecture

## Interactive Game Play On Web

Use `start_backend.sh` to start the backend server.  
* this will start a fastapi server and a redis on docker which stores the game states

Use `start_frontend.sh` to start the frontend server.
* this will install frontend dependencies and start a react app on `http://localhost:3000`

Open `http://localhost:3000` in your browser to play the game.