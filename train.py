import os
import pickle
import logging
from argparse import ArgumentParser

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from prodigyopt import Prodigy
import yaml
import wandb
import random

from MCTS import run_MCTS
from alpha_net_c4 import AlphaLoss, ConnectNet, board_data
from evaluate_arena import parallel_evaluate_net

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the network on")
    parser.add_argument("--config", type=str, default="configs/h5_w5_c3_small.yaml", help="Config file")
    parser.add_argument("--weight_dtype", type=str, default="float32", help="Weight dtype")
    args = parser.parse_args()

    logger.info("Starting iteration pipeline...")
    configs = yaml.safe_load(open(args.config, 'r'))

    os.makedirs("model_ckpts", exist_ok=True)

    wandb.init(project="connect4", config=configs)

    seed_everything()

    initial_step = 0
    model_iteration = 0

    # initialize model
    net = ConnectNet(
        num_cols=configs['board']['num_cols'], 
        num_rows=configs['board']['num_rows'], 
        num_blocks=configs['model']['num_blocks']
    )
    net = net.to(args.device)
    optimizer = Prodigy(net.parameters(), lr=1, slice_p=1, d0=configs['training']['d0'])
    criterion = AlphaLoss()

    weight_dtype = torch.float32 if args.weight_dtype == "float32" else torch.float16
    net = net.to(weight_dtype)

    print(f"Weight dtype: {weight_dtype}")
    print(f"Device: {args.device}")

    pbar = tqdm(range(initial_step, configs['training']['max_train_steps']))
    for global_step in pbar: 
        if global_step % configs['training']['iterate_steps'] == 0:
            # save model
            save_path = f"model_ckpts/{configs['training']['neural_net_name']}_step{global_step}.pth"
            torch.save(net.state_dict(), save_path)
            logger.info(f"Saved model to {save_path}")

            net.eval()

            # evaluate model
            winrate_ai_first, winrate_random_agent = parallel_evaluate_net(net, configs, args.device)
            logs = {"winrate_ai_first": winrate_ai_first, "winrate_random_agent": winrate_random_agent}
            print(logs)
            wandb.log(logs, step=global_step)

            # generate dataset
            run_MCTS(configs, net, start_idx=0, iteration=model_iteration, device=args.device)
            net.train()
            
            data_path = f"./datasets/iter_{model_iteration}/"
            datasets = []
            for idx,file in enumerate(os.listdir(data_path)):
                filename = os.path.join(data_path,file)
                with open(filename, 'rb') as fo:
                    datasets.extend(pickle.load(fo, encoding='bytes'))
            datasets = np.array(datasets, dtype=object)
            train_set = board_data(datasets)
            train_loader = DataLoader(train_set, batch_size=configs['training']['batch_size'], shuffle=True, prefetch_factor=8, num_workers=8, pin_memory=True, persistent_workers=True)

            print(f"Iteration {model_iteration} train set size: {len(train_set)}")
            print(f"Iteration {model_iteration} number of batches: {len(train_loader)}")

            train_iter = iter(train_loader)

            model_iteration += 1

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        state, policy, value = batch
        state, policy, value = state.to(args.device, weight_dtype), policy.to(args.device, weight_dtype), value.to(args.device, weight_dtype)

        policy_pred, value_pred = net(state)

        loss = criterion(value_pred[:,0], value, policy_pred, policy)

        loss.backward()
        clip_grad_norm_(net.parameters(), configs['training']['max_norm'])

        d = optimizer.param_groups[0]['d']
        dlrs = optimizer.param_groups[0]['lr'] * d

        logs = {"loss": loss.detach().item(), "dlrs": dlrs}

        pbar.set_postfix(logs)

        optimizer.step()
        optimizer.zero_grad()

        if global_step % 10 == 0:
            wandb.log(logs, step=global_step)
