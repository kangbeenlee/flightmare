#!/usr/bin/python3.8
from ruamel.yaml import YAML, dump, RoundTripDumper

import os
import sys
import math
import argparse
import numpy as np
import torch
import random
from datetime import datetime

from rpg_baselines.envs import target_tracking_env_wrapper as wrapper
from flightgym import TargetTrackingEnv_v0

from rpg_baselines.single_agent.ddpg import DDPG
from rpg_baselines.single_agent.ddpg import Trainer as DDPGTrainer
from rpg_baselines.single_agent.td3 import TD3
from rpg_baselines.single_agent.td3 import Trainer as TD3Trainer
# from rpg_baselines.single_agent.ppo import PPO
# from rpg_baselines.single_agent.ppo import Trainer as PPOTrainer
from rpg_baselines.single_agent.test import test_model



def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=1, help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)), help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--load_nn', type=str, default='./model/ddpg/actor.pkl', help='Trained actor weight path for ddpg, td3, ppo')
    parser.add_argument("--max_action", type=float, default=3.0, help="Maximum action of actor output")

    # Policy model
    parser.add_argument("--policy", type=str, default="ddpg", help='Policy based reinforcement learning model')
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_z_score_normalization", type=bool, default=False, help="Z score normalization to observation")

    # Learning parameters
    parser.add_argument('--max_training_timesteps', default=int(1e6), type=int, help='Number of training timesteps')
    parser.add_argument('--max_episode_steps', default=1000, type=int, help='Number of steps per episode')
    parser.add_argument('--evaluation_time_steps', default=5000, type=int, help='Number of steps for evaluation')
    parser.add_argument("--evaluation_times", type=int, default=10, help="Evaluate times")
    parser.add_argument('--memory_capacity', default=100000, type=int, help='Replay memory capacity')
    parser.add_argument('--training_start', default=2000, type=int, help='The number of timestep when training start')
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    
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
        hyperparameters.add_argument('--n_envs', default=10, type=int, help="The number of parallel actor")
        hyperparameters.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
        hyperparameters.add_argument("--gae_lambda", default=0.95, type=float, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator")
        hyperparameters.add_argument("--learning_rate", default=3e-4, type=float, help="Actor learning rate")
        hyperparameters.add_argument("--n_epochs", default=15, type=int, help="Update policy for K epochs in one PPO update")
        
        hyperparameters.add_argument("--ent_coef", default=0.0, type=float, help="Weight of entropy loss for exploration")
        hyperparameters.add_argument("--vf_coef", default=0.5, type=float, help="Weight of value loss")
        hyperparameters.add_argument("--max_grad_norm", default=0.5, type=float, help="Maximum gradient norm")
        hyperparameters.add_argument("--clip_range", default=0.2, type=float, help="Clip parameter for PPO")
        hyperparameters.add_argument('--update_timestep', default=256, type=int, help="Update policy every n timesteps")
    else:
        print(f"{args.policy} is unsupported policy")
        sys.exit()
    
    args = parser.parse_args()


    # Set device
    print("============================================================================================")
    if(torch.cuda.is_available()):
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")
    
    # Environment setting parameter
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/target_tracking_env.yaml", 'r'))
    if args.train:
        cfg["env"]["num_envs"] = 1
        if args.policy == "ppo": cfg["env"]["num_envs"] = args.n_envs
        cfg["env"]["num_threads"] = 10
        cfg["env"]["scene_id"] = 0
    else:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 10
        cfg["env"]["num_targets"] = 4
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
            model = DDPG(args,
                         gamma=args.gamma,
                         lr_actor=args.lr_actor,
                         lr_critic=args.lr_critic,
                         tau=args.tau,
                         obs_dim=env.num_obs,
                         action_dim=env.num_acts,
                         max_action=args.max_action)
            trainer = DDPGTrainer(model=model,
                                  env=env,
                                  max_training_timesteps=args.max_training_timesteps,
                                  max_episode_steps=args.max_episode_steps,
                                  evaluation_time_steps=args.evaluation_time_steps,
                                  evaluation_times=args.evaluation_times,
                                  obs_dim=env.num_obs,
                                  action_dim=env.num_acts,
                                  max_action=args.max_action,
                                  expl_noise=args.expl_noise,
                                  memory_capacity=args.memory_capacity,
                                  batch_size=args.batch_size,
                                  training_start=args.training_start,
                                  save_dir=args.save_dir)
        elif args.policy == "td3":
            model = TD3(args,
                        gamma=args.gamma,
                        lr_actor=args.lr_actor,
                        lr_critic=args.lr_critic,
                        tau=args.tau,                        
                        policy_noise=args.policy_noise, # std
                        noise_clip=args.noise_clip,
                        policy_freq=args.policy_freq,                        
                        obs_dim=env.num_obs,
                        action_dim=env.num_acts,
                        max_action=args.max_action)
            trainer = TD3Trainer(model=model,
                                 env=env,
                                 max_training_timesteps=args.max_training_timesteps,
                                 max_episode_steps=args.max_episode_steps,
                                 evaluation_time_steps=args.evaluation_time_steps,
                                 evaluation_times=args.evaluation_times,
                                 obs_dim=env.num_obs,
                                 action_dim=env.num_acts,
                                 max_action=args.max_action,
                                 expl_noise=args.expl_noise,
                                 memory_capacity=args.memory_capacity,
                                 batch_size=args.batch_size,
                                 training_start=args.training_start,
                                 save_dir=args.save_dir)
        # elif args.policy == "ppo":
        #     model = PPO(n_envs=args.n_envs,
        #                 gamma=args.gamma,
        #                 gae_lambda=args.gae_lambda,
        #                 rollout_length=args.update_timestep,
        #                 learning_rate=args.learning_rate,
        #                 n_epochs=args.n_epochs,
        #                 ent_coef=args.ent_coef,
        #                 vf_coef=args.vf_coef,
        #                 max_grad_norm=args.max_grad_norm,
        #                 batch_size=64,
        #                 clip_range=args.clip_range,
        #                 obs_dim=env.num_obs,
        #                 action_dim=env.num_acts,
        #                 max_action=args.max_action)
        #     trainer = PPOTrainer(model=model,
        #                          env=env,
        #                          n_envs=1,
        #                          max_training_timesteps=args.max_training_timesteps,
        #                          max_episode_steps=args.max_episode_steps,
        #                          evaluation_time_steps=args.evaluation_time_steps,
        #                          evaluation_times=args.evaluation_times,
        #                          obs_dim=env.num_obs,
        #                          action_dim=env.num_acts,
        #                          max_action=args.max_action,
        #                          save_dir=args.save_dir)
        else:
            print(f"{args.policy} is unsupported policy")
            sys.exit()
        
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
            model = DDPG(args,
                         obs_dim=env.num_obs,
                         action_dim=env.num_acts)
            model.load(args.load_nn)
            test_model(env, model=model, render=args.render, max_episode_steps=args.max_episode_steps)
        elif args.policy == "td3":
            model = TD3(args,
                        obs_dim=env.num_obs,
                        action_dim=env.num_acts)
            model.load(args.load_nn)
            test_model(env, model=model, render=args.render, max_episode_steps=args.max_episode_steps)
        # elif args.policy == "ppo":
        #     model = PPO(obs_dim=env.num_obs,
        #                 action_dim=env.num_acts)
        #     model.load(args.load_nn)
        #     test_model(env, model=model, render=args.render, max_episode_steps=args.max_episode_steps)
        else:
            print(f"{args.policy} is unsupported policy")
            sys.exit()
        
if __name__ == "__main__":
    main()