# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 10:11:36 2022

@author: Utilisateur
"""
import argparse
import os
import sys
import csv
import socket
import numpy as np


from simulation_base.env import resume_env, nb_actuations
from RemoteEnvironmentClient import RemoteEnvironmentClient


from multiprocessing import Process, Lock
from experiment_parallel import experiment


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
    parser.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
    parser.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)
    
    parser.add_argument('--env', type=str, default='Cylinder2D')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    
    parser.add_argument('--K', type=int, default=20)
    
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    
    parser.add_argument('--embed_dim', type=int, default=128)
    
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=1)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)
    
    parser.add_argument('--max_ep_len', type=int, default=500)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--env_targets', type=list, default=[0.1])
    
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    
    arguments = vars(parser.parse_args())
    
    
    number_servers = arguments["number_servers"]
    ports_start = arguments["ports_start"]
    host = arguments["host"]

    if host == 'None':
        host = socket.gethostname()
    
    example_environment = resume_env(plot=False, dump_CL=100, dump_debug=1, dump_vtu=50)

    environments = []
    for crrt_simu in range(number_servers):
        environments.append(RemoteEnvironmentClient(
            example_environment, verbose=0, port=ports_start + crrt_simu, host=host,
            timing_print=(crrt_simu == 0)     # Only print time info for env_0
        ))

    max_ep_len = arguments["max_ep_len"]
    scale = arguments["scale"]
    env_targets = arguments["env_targets"]
    
    for env in environments:
        p = Process(target = experiment, args = (arguments,env, max_ep_len,env_targets,scale))
        p.start()
        
    for env in environments:
        env.close()
