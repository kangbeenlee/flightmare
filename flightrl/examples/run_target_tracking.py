#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

import os
import math
import argparse
import numpy as np
import torch
import random

from rpg_baselines.envs import target_tracking_env_wrapper as wrapper
from flightgym import TargetTrackingEnv_v0

from rpg_baselines.ddpg.ddpg import Trainer
# from rpg_baselines.ddpg.ddpg_test import test_model
from rpg_baselines.ddpg.target_test import test_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1, help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=1, help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)), help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    # parser.add_argument('--load_nn', type=str, default='./saved/quadrotor_env.zip', help='trained weight path')
    parser.add_argument('--load_nn', type=str, default='./saved', help='Trained weight path')

    # Deep Recurrent Q-network
    parser.add_argument('--total_timesteps', default=25000000, type=int, help='Number of training episode (epochs)')
    parser.add_argument('--rollout_steps', default=250, type=int, help='Number of steps per episode')
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Training parameters
    parser.add_argument("--lr_q", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lr_mu", type=float, default=5e-4, help="Learning rate")

    # Target network update
    parser.add_argument('--training_start', default=2000, type=int, help='Batch size')
    parser.add_argument("--tau", type=float, default=0.005, help="Soft target update parameter")
    parser.add_argument("--use_hard_update", action="store_true", help="Use hard target network update")
    parser.add_argument("--target_update_period", type=int, default=300, help="Target network update period")

    # Experience replay
    parser.add_argument('--memory_capacity', default=50000, type=int, help='Replay memory capacity')
    
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("------ Use {} ------".format(args.device))
    
    
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/target_tracking_env.yaml", 'r'))
    cfg_quadrotor = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/custom_quadrotor_env.yaml", 'r'))

    # if not args.train:
    #     cfg["env"]["num_envs"] = 1
    #     cfg["env"]["num_threads"] = 1
    # if args.render:
    #     cfg["env"]["render"] = "yes"
    # else:
    #     cfg["env"]["render"] = "no"
        
    cfg["env"]["num_envs"] = 1
    cfg["env"]["num_threads"] = 1
    cfg["env"]["render"] = "yes"

    cfg_quadrotor["quadrotor_pid_controller_gain"]["kp_vxy"] = 5.0
    cfg_quadrotor["quadrotor_pid_controller_gain"]["ki_vxy"] = 0.0
    cfg_quadrotor["quadrotor_pid_controller_gain"]["kd_vxy"] = 0.0
    cfg_quadrotor["quadrotor_pid_controller_gain"]["kp_vz"] = 4.0
    cfg_quadrotor["quadrotor_pid_controller_gain"]["ki_vz"] = 0.0
    cfg_quadrotor["quadrotor_pid_controller_gain"]["kd_vz"] = 0.3
    cfg_quadrotor["quadrotor_pid_controller_gain"]["kp_angle"] = 5.0
    cfg_quadrotor["quadrotor_pid_controller_gain"]["ki_angle"] = 0
    cfg_quadrotor["quadrotor_pid_controller_gain"]["kd_angle"] = 1.5
    cfg_quadrotor["quadrotor_pid_controller_gain"]["kp_wz"] = 4.0
    cfg_quadrotor["quadrotor_pid_controller_gain"]["ki_wz"] = 0.0
    cfg_quadrotor["quadrotor_pid_controller_gain"]["kd_wz"] = 0.0

    env = wrapper.FlightmareTargetTrackingEnv(TargetTrackingEnv_v0(dump(cfg, Dumper=RoundTripDumper), dump(cfg_quadrotor, Dumper=RoundTripDumper), False))

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)

    test_model(env, render=args.render)        

if __name__ == "__main__":
    main()