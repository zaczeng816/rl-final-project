# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:01:30 2019

@author: WT
"""
from MCTS_c4 import run_MCTS
from train_c4 import train_connectnet
from evaluator_c4 import evaluate_nets
from argparse import ArgumentParser
import logging
import yaml

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=1000, help="Total number of iterations to run")
    parser.add_argument("--config", type=str, default="configs/h5_w5_c3_small.yaml", help="Config file")
    args = parser.parse_args()
    
    logger.info("Starting iteration pipeline...")
    configs = yaml.safe_load(open(args.config, 'r'))

    for i in range(args.iteration, args.total_iterations): 
        run_MCTS(configs, start_idx=0, iteration=i)
        train_connectnet(configs, iteration=i, new_optim_state=True)
        if i >= 1:
            winner = evaluate_nets(configs, i, i + 1)
            counts = 0
            while (winner != (i + 1)):
                logger.info("Trained net didn't perform better, generating more MCTS games for retraining...")
                run_MCTS(configs, start_idx=(counts + 1)*configs['self_play']['num_games_per_MCTS_process'], iteration=i)
                counts += 1
                train_connectnet(configs, iteration=i, new_optim_state=True)
                winner = evaluate_nets(configs, i, i + 1)