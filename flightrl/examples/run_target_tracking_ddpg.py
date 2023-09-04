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

from rpg_baselines.ddpg.ddpg import Trainer, DDPG
from rpg_baselines.ddpg.ddpg_test import test_model
# from rpg_baselines.ddpg.control_test import test_model


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=0, help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=1, help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)), help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--load_nn_actor', type=str, default='./saved/ddpg_actor.pkl', help='Trained actor weight path')
    parser.add_argument('--load_nn_critic', type=str, default='./saved/ddpg_critic.pkl', help='Trained critic weight path')

    # Training parameters
    parser.add_argument('--num_episodes', default=5000, type=int, help='Number of training episode (epochs)')
    parser.add_argument('--max_episode_steps', default=300, type=int, help='Number of steps per episode')
    parser.add_argument('--memory_capacity', default=100000, type=int, help='Replay memory capacity')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--training_start', default=2000, type=int, help='Batch size')
    
    # Model hyperparameters
    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    hyperparameters.add_argument("--lr_actor", type=float, default=5e-4, help="Actor learning rate")
    hyperparameters.add_argument("--lr_critic", type=float, default=0.001, help="Critic learning rate")
    hyperparameters.add_argument("--tau", type=float, default=0.005, help="Soft target update parameter")
    hyperparameters.add_argument("--use_hard_update", action="store_true", help="Use hard target network update")
    hyperparameters.add_argument("--target_update_period", type=int, default=300, help="Target network update period for hard target network update")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("------ Use {} ------".format(args.device))
    
    
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/target_tracking_env.yaml", 'r'))

    if args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1
        cfg["env"]["scene_id"] = 0
    else:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1  
        cfg["env"]["scene_id"] = 0      
    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    # Generate target tracking environment        
    env = wrapper.FlightmareTargetTrackingEnv(TargetTrackingEnv_v0(dump(cfg, Dumper=RoundTripDumper), False))
    
    # Set random seed
    configure_random_seed(args.seed, env=env)

    if args.train:
        model = DDPG(device=args.device,
                     gamma=args.gamma,
                     lr_actor=args.lr_actor,
                     lr_critic=args.lr_critic,
                     tau=args.tau,
                     use_hard_update=args.use_hard_update,
                     target_update_period=args.target_update_period,
                     obs_dim=env.num_obs,
                     action_dim=env.num_acts - 1)
        
        trainer = Trainer(model=model,
                          env=env,
                          num_episodes=args.num_episodes,
                          max_episode_steps=args.max_episode_steps,
                          obs_dim=env.num_obs,
                          action_dim=env.num_acts - 1,
                          memory_capacity=args.memory_capacity,
                          batch_size=args.batch_size,
                          training_start=args.training_start,
                          save_dir=args.save_dir)
        trainer.learn(render=args.render)
    else:
        # Load trained model!
        model = DDPG(obs_dim=env.num_obs,
                     action_dim=env.num_acts - 1)
        model.load_models(args.load_nn_actor, args.load_nn_critic)
        test_model(env, model=model, render=args.render, max_episode_steps=args.max_episode_steps)
        # test_model(env, render=args.render)

if __name__ == "__main__":
    main()