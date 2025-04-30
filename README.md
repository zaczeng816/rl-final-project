# rl-final-project

## Setup

```bash
conda create -n rl-final python==3.10
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
