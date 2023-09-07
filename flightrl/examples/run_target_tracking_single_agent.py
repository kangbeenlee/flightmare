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

from rpg_baselines.single_agent.ddpg.ddpg import DDPG
from rpg_baselines.single_agent.ddpg.ddpg import Trainer as DDPGTrainer
from rpg_baselines.single_agent.td3.td3 import TD3
from rpg_baselines.single_agent.td3.td3 import Trainer as TD3Trainer
# from rpg_baselines.single_agent.ppo.ppo import PPO, 
# from rpg_baselines.single_agent.ppo.ppo import Trainer as PPOTrainer
from rpg_baselines.single_agent.test import test_model
# from rpg_baselines.single_agent.test_control import test_model



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

    # Policy model
    parser.add_argument("--policy", type=str, default="ddpg", help='Policy based reinforcement learning model')

    # Learning parameters
    parser.add_argument('--num_episodes', default=1000, type=int, help='Number of training episode (epochs)')
    parser.add_argument('--max_episode_steps', default=300, type=int, help='Number of steps per episode')
    parser.add_argument('--memory_capacity', default=100000, type=int, help='Replay memory capacity')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--training_start', default=3000, type=int, help='The number of timestep when training start')
    
    args = parser.parse_args()
    
    # Model hyperparameters
    if args.policy == "ddpg":
        hyperparameters = parser.add_argument_group('ddpg_hyperparameters')
        hyperparameters.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
        hyperparameters.add_argument("--lr_actor", type=float, default=3e-4, help="Actor learning rate")
        hyperparameters.add_argument("--lr_critic", type=float, default=3e-4, help="Critic learning rate")
        hyperparameters.add_argument("--tau", type=float, default=0.005, help="Soft target update parameter")
    elif args.policy == "td3":
        hyperparameters = parser.add_argument_group('td3_hyperparameters')
        hyperparameters.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
        hyperparameters.add_argument("--lr_actor", default=3e-4, type=float, help="Actor learning rate")
        hyperparameters.add_argument("--lr_critic", default=3e-4, type=float, help="Critic learning rate")
        hyperparameters.add_argument("--tau", type=float, default=0.005, help="Soft target update parameter")
        hyperparameters.add_argument("--expl_noise", default=0.1, type=float, help="Std of Gaussian exploration noise")
        hyperparameters.add_argument("--policy_noise", default=0.2, type=float, help="Noise added to target policy during critic update")
        hyperparameters.add_argument("--noise_clip", default=0.5, type=float, help="Range to clip target policy noise")
        hyperparameters.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")
    else:
        hyperparameters = parser.add_argument_group('ppo_hyperparameters')
        hyperparameters.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
        hyperparameters.add_argument("--lr_actor", default=3e-4, type=float, help="Actor learning rate")
        hyperparameters.add_argument("--lr_critic", default=3e-4, type=float, help="Critic learning rate")
        hyperparameters.add_argument("--expl_noise", default=0.1, type=float, help="Std of Gaussian exploration noise")
        hyperparameters.add_argument("--policy_noise", default=0.2, type=float, help="Noise added to target policy during critic update")
        hyperparameters.add_argument("--noise_clip", default=0.5, type=float, help="Range to clip target policy noise")
        hyperparameters.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")

    args = parser.parse_args()

    #
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Flightrl Single Agent Reinforcement Learning Environment")
    print(f">>> Device: {args.device}, Policy: {args.policy}")
    
    #
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/target_tracking_env.yaml", 'r'))

    #
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
        if args.policy == "ddpg":
            model = DDPG(device=args.device,
                        gamma=args.gamma,
                        lr_actor=args.lr_actor,
                        lr_critic=args.lr_critic,
                        tau=args.tau,
                        obs_dim=env.num_obs,
                        action_dim=env.num_acts - 1,
                        max_action=3.0)
            trainer = DDPGTrainer(model=model,
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
        
        elif args.policy == "td3":
            model = TD3(device=args.device,
                        gamma=args.gamma,
                        lr_actor=args.lr_actor,
                        lr_critic=args.lr_critic,
                        tau=args.tau,                        
                        policy_noise=args.policy_noise, # std
                        noise_clip=args.noise_clip,
                        policy_freq=args.policy_freq,                        
                        obs_dim=env.num_obs,
                        action_dim=env.num_acts - 1,
                        max_action=3.0)
            trainer = TD3Trainer(model=model,
                                 env=env,
                                 num_episodes=args.num_episodes,
                                 max_episode_steps=args.max_episode_steps,
                                 obs_dim=env.num_obs,
                                 action_dim=env.num_acts - 1,
                                 max_action=3.0,
                                 expl_noise=args.expl_noise,
                                 memory_capacity=args.memory_capacity,
                                 batch_size=args.batch_size,
                                 training_start=args.training_start,
                                 save_dir=args.save_dir)
            trainer.learn(render=args.render)
        
        else:
            print("ppo")
            # model = PPO()
            # trainer = PPOTrainer()
            
    else:
        # Load trained model!
        if args.policy == "ddpg":
            model = DDPG(obs_dim=env.num_obs,
                         action_dim=env.num_acts -1)
            model.load(args.load_nn_actor, args.load_nn_critic)
            test_model(env, model=model, render=args.render, max_episode_steps=args.max_episode_steps)
            # test_model(env, render=args.render)

        elif args.policy == "td3":
            model = TD3(obs_dim=env.num_obs,
                        action_dim=env.num_acts)
            model.load(args.load_nn_actor, args.load_nn_critic)
            test_model(env, model=model, render=args.render, max_episode_steps=args.max_episode_steps)
            
        else:
            print("ppo")
        
if __name__ == "__main__":
    main()