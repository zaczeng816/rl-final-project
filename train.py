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
import yaml
import wandb

from MCTS import run_MCTS
from alpha_net_c4 import AlphaLoss, ConnectNet, board_data

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=50000, help="Number of steps to train the network")
    parser.add_argument("--iterate_steps", type=int, default=500, help="Number of steps to iterate the network")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the network on")
    parser.add_argument("--config", type=str, default="configs/h6_w7_c4_small.yaml", help="Config file")
    parser.add_argument("--weight_dtype", type=str, default="float32", help="Weight dtype")
    args = parser.parse_args()

    logger.info("Starting iteration pipeline...")
    configs = yaml.safe_load(open(args.config, 'r'))

    os.makedirs("model_ckpts", exist_ok=True)

    wandb.init(project="connect4", config=configs)

    initial_step = 0
    model_iteration = 0

    # initialize model
    net = ConnectNet(
        num_cols=configs['board']['num_cols'], 
        num_rows=configs['board']['num_rows'], 
        num_blocks=configs['model']['num_blocks']
    )
    net = net.to(args.device)
    optimizer = optim.Adam(net.parameters(), lr=configs['training']['lr'], betas=(0.8, 0.999))
    criterion = AlphaLoss()

    weight_dtype = torch.float32 if args.weight_dtype == "float32" else torch.float16
    net = net.to(weight_dtype)

    print(f"Weight dtype: {weight_dtype}")
    print(f"Device: {args.device}")

    pbar = tqdm(range(initial_step, args.train_steps))
    for global_step in pbar: 
        if global_step % args.iterate_steps == 0:
            # save model
            torch.save(net.state_dict(), f"model_ckpts/{configs['training']['neural_net_name']}_step{global_step}.pth")

            # generate dataset
            net.eval()
            run_MCTS(configs, net, start_idx=0, iteration=model_iteration)

            data_path="./datasets/iter_%d/" % model_iteration
            datasets = []
            for idx,file in enumerate(os.listdir(data_path)):
                filename = os.path.join(data_path,file)
                with open(filename, 'rb') as fo:
                    datasets.extend(pickle.load(fo, encoding='bytes'))
            datasets = np.array(datasets, dtype=object)
            train_set = board_data(datasets)
            train_loader = DataLoader(train_set, batch_size=configs['training']['batch_size'], shuffle=True, num_workers=8, pin_memory=False)

            model_iteration += 1
            net.train()
        
        batch = next(iter(train_loader))
        state, policy, value = batch
        state, policy, value = state.to(args.device, weight_dtype), policy.to(args.device, weight_dtype), value.to(args.device, weight_dtype)

        policy_pred, value_pred = net(state)

        loss = criterion(value_pred[:,0], value, policy_pred, policy)

        loss.backward()
        clip_grad_norm_(net.parameters(), configs['training']['max_norm'])

        logs = {"loss": loss.detach().item()}

        pbar.set_postfix(logs)

        optimizer.step()
        optimizer.zero_grad()

        wandb.log(logs, step=global_step)
