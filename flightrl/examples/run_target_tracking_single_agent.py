#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

import os
import math
import argparse
import numpy as np
import torch
import random
from datetime import datetime

from rpg_baselines.envs import target_tracking_env_wrapper as wrapper
from flightgym import TargetTrackingEnv_v0

from rpg_baselines.single_agent.ddpg.ddpg import DDPG
from rpg_baselines.single_agent.ddpg.ddpg import Trainer as DDPGTrainer
from rpg_baselines.single_agent.td3.td3 import TD3
from rpg_baselines.single_agent.td3.td3 import Trainer as TD3Trainer
from rpg_baselines.single_agent.ppo.ppo import PPO
from rpg_baselines.single_agent.ppo.ppo import Trainer as PPOTrainer
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
    parser.add_argument('--load_nn_actor', type=str, default='./saved/actor.pkl', help='Trained actor weight path for ddpg and td3')
    parser.add_argument('--load_nn_critic', type=str, default='./saved/critic.pkl', help='Trained critic weight path for ddpg and td3')
    parser.add_argument('--load_nn_actor_critic', type=str, default='./saved/actor_critic.pkl', help='Trained actor critic weight path for ppo')

    # Policy model
    parser.add_argument("--policy", type=str, default="ddpg", help='Policy based reinforcement learning model')

    # Learning parameters
    parser.add_argument('--max_training_timesteps', default=1000000, type=int, help='Number of training timesteps')
    parser.add_argument('--max_episode_steps', default=300, type=int, help='Number of steps per episode')
    parser.add_argument('--evaluation_time_steps', default=5000, type=int, help='Number of steps for evaluation')
    parser.add_argument('--memory_capacity', default=100000, type=int, help='Replay memory capacity')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--training_start', default=2000, type=int, help='The number of timestep when training start')
    
    args = parser.parse_args()
    
    # Model hyperparameters
    if args.policy == "ddpg":
        hyperparameters = parser.add_argument_group('ddpg_hyperparameters')
        hyperparameters.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
        hyperparameters.add_argument("--lr_actor", type=float, default=3e-4, help="Actor learning rate")
        hyperparameters.add_argument("--lr_critic", type=float, default=3e-4, help="Critic learning rate")
        hyperparameters.add_argument("--tau", type=float, default=0.005, help="Soft target update parameter")
        hyperparameters.add_argument("--expl_noise", default=0.1, type=float, help="Std of Gaussian exploration noise")
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
    elif args.policy == "ppo":
        hyperparameters = parser.add_argument_group('ppo_hyperparameters')
        hyperparameters.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
        hyperparameters.add_argument("--lr_actor", default=3e-4, type=float, help="Actor learning rate")
        hyperparameters.add_argument("--lr_critic", default=0.001, type=float, help="Critic learning rate")
        hyperparameters.add_argument("--K_epochs", default=80, type=int, help="Update policy for K epochs in one PPO update")
        hyperparameters.add_argument("--eps_clip", default=0.2, type=float, help="Clip parameter for PPO")
        hyperparameters.add_argument("--action_std_init", default=0.6, type=float, help="Starting std for action distribution (Multivariate Normal)")
        hyperparameters.add_argument("--action_std_decay_rate", default=0.05, type=float, help="Linearly decay action_std (action_std = action_std - action_std_decay_rate)")
        hyperparameters.add_argument("--min_action_std", default=0.1, type=float, help="Minimum action_std (stop decay after action_std <= min_action_std)")
        hyperparameters.add_argument('--action_std_decay_freq', default=100000, type=int, help="Action_std decay frequency (in num timesteps)")
        hyperparameters.add_argument('--update_timestep', default=2000, type=int, help="Update policy every n timesteps")
    else:
        print(f"{args.policy} is unsupported policy")
        os.exit()
    
    args = parser.parse_args()


    # Set device
    print("============================================================================================")
    args.device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        args.device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(args.device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")
    
    # Environment setting parameter
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

    # Environment and policy type information
    print("Training environment name : Flightrl Single Agent Reinforcement Learning Environment")
    print("Scene ID :", cfg["env"]["scene_id"])
    if args.train: print("Policy to be trained :", args.policy)
    else: print("Policy to be tested :", args.policy)
    print("The number of tracker (agent) :", cfg["env"]["num_envs"])
    print("--------------------------------------------------------------------------------------------")

    # Generate target tracking environment
    env = wrapper.FlightmareTargetTrackingEnv(TargetTrackingEnv_v0(dump(cfg, Dumper=RoundTripDumper), False))
    configure_random_seed(args.seed, env=env)

    print("--------------------------------------------------------------------------------------------")
    print("Max training timesteps :", args.max_training_timesteps)
    print("Max timesteps per episode :", args.max_episode_steps)
    print("Evaluation timesteps :", args.evaluation_time_steps)
    print("--------------------------------------------------------------------------------------------")
    print("Observation space dimension :", env.num_obs)
    print("Action space dimension :", env.num_acts)
    print("--------------------------------------------------------------------------------------------")


    if args.train:
        ########################### Model definition ###########################
        if args.policy == "ddpg":
            model = DDPG(device=args.device,
                         gamma=args.gamma,
                         lr_actor=args.lr_actor,
                         lr_critic=args.lr_critic,
                         tau=args.tau,
                         obs_dim=env.num_obs,
                         action_dim=env.num_acts,
                         max_action=3.0)
            trainer = DDPGTrainer(model=model,
                                  env=env,
                                  max_training_timesteps=args.max_training_timesteps,
                                  max_episode_steps=args.max_episode_steps,
                                  evaluation_time_steps=args.evaluation_time_steps,
                                  obs_dim=env.num_obs,
                                  action_dim=env.num_acts,
                                  max_action=3.0,
                                  expl_noise=args.expl_noise,
                                  memory_capacity=args.memory_capacity,
                                  batch_size=args.batch_size,
                                  training_start=args.training_start,
                                  save_dir=args.save_dir)
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
                        action_dim=env.num_acts,
                        max_action=3.0)
            trainer = TD3Trainer(model=model,
                                 env=env,
                                 max_training_timesteps=args.max_training_timesteps,
                                 max_episode_steps=args.max_episode_steps,
                                 evaluation_time_steps=args.evaluation_time_steps,
                                 obs_dim=env.num_obs,
                                 action_dim=env.num_acts,
                                 max_action=3.0,
                                 expl_noise=args.expl_noise,
                                 memory_capacity=args.memory_capacity,
                                 batch_size=args.batch_size,
                                 training_start=args.training_start,
                                 save_dir=args.save_dir)
        elif args.policy == "ppo":
            model = PPO(device=args.device,
                        gamma=args.gamma,
                        obs_dim=env.num_obs,
                        action_dim=env.num_acts,
                        lr_actor=args.lr_actor,
                        lr_critic=args.lr_critic,
                        K_epochs=args.K_epochs,
                        eps_clip=args.eps_clip,
                        action_std_init=args.action_std_init)
            trainer = PPOTrainer(model=model,
                                 env=env,
                                 max_training_timesteps=args.max_training_timesteps,
                                 max_episode_steps=args.max_episode_steps,
                                 evaluation_time_steps=args.evaluation_time_steps,
                                 update_timestep=args.update_timestep,
                                 action_std_decay_freq=args.action_std_decay_freq,
                                 action_std_decay_rate=args.action_std_decay_rate,
                                 min_action_std=args.min_action_std,
                                 save_dir=args.save_dir)
        else:
            print(f"{args.policy} is unsupported policy")
            os.exit()
        
        ########################### Training section ###########################
        start_time = datetime.now().replace(microsecond=0)
        print("============================================================================================")
        print("Started training at (GMT) : ", start_time)
        print("============================================================================================")

        trainer.learn(render=args.render)

        end_time = datetime.now().replace(microsecond=0)
        print("============================================================================================")
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")
    
    else:
        # Load trained model!
        if args.policy == "ddpg":
            model = DDPG(device=args.device,
                         obs_dim=env.num_obs,
                         action_dim=env.num_acts)
            model.load(args.load_nn_actor, args.load_nn_critic)
            test_model(env, model=model, render=args.render, max_episode_steps=args.max_episode_steps)
            # test_model(env, render=args.render)
        elif args.policy == "td3":
            model = TD3(device=args.device,
                        obs_dim=env.num_obs,
                        action_dim=env.num_acts)
            model.load(args.load_nn_actor, args.load_nn_critic)
            test_model(env, model=model, render=args.render, max_episode_steps=args.max_episode_steps)
            # test_model(env, render=args.render)
        elif args.policy == "ppo":
            model = PPO(device=args.device,
                        obs_dim=env.num_obs,
                        action_dim=env.num_acts)
            model.load(args.load_nn_actor_critic)
            test_model(env, model=model, render=args.render, max_episode_steps=args.max_episode_steps)
            # test_model(env, render=args.render)
        else:
            print(f"{args.policy} is unsupported policy")
            os.exit()
        
if __name__ == "__main__":
    main()