#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper
from rpg_baselines.envs import target_tracking_env_wrapper as wrapper
from flightgym import TargetTrackingEnv_v0

import os
import sys
import random
import copy
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# from make_env import make_env
import argparse
from rpg_baselines.multi_agent.replay_buffer import ReplayBuffer
from rpg_baselines.multi_agent.maddpg import MADDPG
from rpg_baselines.multi_agent.matd3 import MATD3
from rpg_baselines.multi_agent.test import test_model





class Runner:
    def __init__(self, args, env):
        self.args = args
        self.render = args.render        
        self.env = env
        self.N = args.N

        # Create N agents
        if self.args.policy == "maddpg":
            self.agent_n = [MADDPG(args, agent_id) for agent_id in range(self.N)]
        elif self.args.policy == "matd3":
            self.agent_n = [MATD3(args, agent_id) for agent_id in range(self.N)]
        else:
            print(f"{self.args.policy} is not supported marl policy")
            sys.exit()

        if self.args.load_weight:
            print("--------------------------------------------------------------------------------------------")
            load_nn = os.path.join(args.load_nn, 'batch_{}_{}'.format(args.batch_size, args.policy))
            for agent_id in range(args.N):
                self.agent_n[agent_id].load_model(os.path.join(load_nn, 'actor_{}_{}k.pth'.format(agent_id, args.iteration)))
            print("--------------------------------------------------------------------------------------------")

        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        # self.writer = SummaryWriter(log_dir='runs/multi/batch_{}_{}'.format(self.args.batch_size, self.args.policy))
        self.writer = SummaryWriter(log_dir='runs/multi/multi_ctbr'.format(self.args.batch_size, self.args.policy))
        # Total training time step
        self.time_steps = 0
        # TQDM training bar
        self.tqdm_bar = tqdm(initial=0, desc="Training", total=self.args.max_training_timesteps, unit="timestep", dynamic_ncols=True)
        # Initialize noise_std
        self.noise_std = self.args.noise_std_init

        self.best_score = None
        self.best_step = None

    def run(self, ):
        if self.render:
            self.env.connectUnity()

        self.evaluate_policy()

        while self.time_steps < self.args.max_training_timesteps:
            obs_n = self.env.reset()
            
            for _ in range(self.args.max_episode_steps):
                self.tqdm_bar.update(1)
                
                # Multi agent selects actions based on its own local observations (add noise for exploration)
                a_n = np.array([agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, obs_n)]).astype(np.float32)
                obs_next_n, r_n, done_n, _ = self.env.step(copy.deepcopy(a_n))

                # Store the transition
                self.replay_buffer.store_transition(obs_n, a_n, r_n, obs_next_n, done_n)
                obs_n = obs_next_n
                self.time_steps += 1

                # Decay noise_std
                if self.args.use_noise_decay:
                    self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                if self.replay_buffer.current_size > self.args.batch_size:
                    # Train each agent individually
                    for agent_id in range(self.N):
                        self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)

                if self.time_steps % self.args.evaluation_time_steps == 0:
                    self.evaluate_policy()

                if all(done_n):
                    break

        if self.render:
            self.env.disconnectUnity()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluation_times):
            obs_n = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.args.max_episode_steps):
                a_n = np.array([agent.choose_action(obs, noise_std=0) for agent, obs in zip(self.agent_n, obs_n)]).astype(np.float32)
                obs_next_n, r_n, done_n, _ = self.env.step(copy.deepcopy(a_n))
                episode_reward += np.mean(r_n)
                # episode_reward += r_n[0] # All agent get same team reward
                obs_n = obs_next_n
                
                if any(done_n):
                    break
            
            evaluate_reward += episode_reward
        evaluate_reward = evaluate_reward / self.args.evaluation_times
        
        # Save best model
        # save_path = os.path.join('./model', 'batch_{}_{}'.format(self.args.batch_size, self.args.policy))
        save_path = os.path.join('./model', 'multi_ctbr'.format(self.args.batch_size, self.args.policy))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.best_score == None or evaluate_reward > self.best_score:
            self.best_score = evaluate_reward
            self.best_step  = self.time_steps
            # Save the rewards and models
            for agent_id in range(self.N):
                self.agent_n[agent_id].save_model(save_path, agent_id, self.time_steps)

        print("time_steps:{} \t\t evaluate_reward:{} \t\t noise_std:{} \t\t best step:{}".format(self.time_steps, evaluate_reward, self.noise_std, self.best_step))
        self.writer.add_scalar("evaluate_step_rewards", evaluate_reward, global_step=self.time_steps)


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3")

    parser.add_argument('--n', type=int, default=3, help="Number of agent (tracker)")
    parser.add_argument('--n_targets', type=int, default=1, help="Number of target")
    parser.add_argument('--train', action="store_true", help="To train new model or simply test pre-trained model")
    parser.add_argument('--load_weight', action="store_true", help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=1, help="Enable Unity Render")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--load_nn', type=str, default='./model', help='Trained actor weight path for ddpg and td3')
    parser.add_argument('--iteration', type=str, default='0', help='Choose trained iteration')
    parser.add_argument('--gpu_id', type=str, default='cuda:0', help='Choose gpu device id')

    parser.add_argument("--max_training_timesteps", type=int, default=int(5e6), help=" Maximum number of training steps")
    parser.add_argument("--max_episode_steps", type=int, default=200, help="Maximum number of steps per episode")
    parser.add_argument("--evaluation_time_steps", type=float, default=5000, help="Evaluate the policy every 'evaluation_time_steps'")
    parser.add_argument("--evaluation_times", type=float, default=5, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=3.0, help="Max action")

    parser.add_argument("--policy", type=str, default="maddpg", help="maddpg or matd3")
    parser.add_argument("--buffer_size", type=int, default=int(5e5), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--actor_hidden_dim", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--critic_hidden_dim", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5, help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    # parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    # parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate of critic")

    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")

    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    # Set device
    print("============================================================================================")
    args.device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        args.device = torch.device(args.gpu_id)
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(args.device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")

    # Environment setting parameter
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/target_tracking_env.yaml", 'r'))
    if args.train:
        cfg["env"]["num_envs"] = args.n
        cfg["env"]["num_threads"] = 10
        cfg["env"]["num_targets"] = args.n_targets
        cfg["env"]["scene_id"] = 0
    else:
        cfg["env"]["num_envs"] = args.n
        cfg["env"]["num_threads"] = 1
        cfg["env"]["num_targets"] = args.n_targets
        cfg["env"]["scene_id"] = 0
    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    # Environment and policy type information
    print("Training environment name : Flightrl Multi Agent Reinforcement Learning Environment")
    print("Scene ID :", cfg["env"]["scene_id"])
    if args.train: print("Policy to be trained :", args.policy)
    else: print("Policy to be tested :", args.policy)
    print("The number of tracker (agent) :", cfg["env"]["num_envs"])
    print("--------------------------------------------------------------------------------------------")

    # Generate target tracking environment
    env = wrapper.FlightmareTargetTrackingEnv(TargetTrackingEnv_v0(dump(cfg, Dumper=RoundTripDumper), False))
    configure_random_seed(args.seed, env=env)

    args.N = env.num_envs # The number of agents
    args.obs_dim = env.num_obs
    args.action_dim = env.num_acts
    args.critic_input_dim = (args.obs_dim + args.action_dim) * args.N

    print("--------------------------------------------------------------------------------------------")
    print("Max training timesteps :", args.max_training_timesteps)
    print("Max timesteps per episode :", args.max_episode_steps)
    print("Evaluation timesteps :", args.evaluation_time_steps)
    print("--------------------------------------------------------------------------------------------")
    print("Critic input space dimension :", args.critic_input_dim)
    print("Observation space dimension :", env.num_obs)
    print("Action space dimension :", env.num_acts)
    print("--------------------------------------------------------------------------------------------")

    if args.train:
        runner = Runner(args, env)
        start_time = datetime.now().replace(microsecond=0)
        print("============================================================================================")
        print("Started training at (GMT) : ", start_time)
        print("============================================================================================")
        runner.run()
        end_time = datetime.now().replace(microsecond=0)
        print("============================================================================================")
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")
    else:
        # Load trained model!
        load_nn = os.path.join(args.load_nn, 'batch_{}_{}'.format(args.batch_size, args.policy))

        if args.policy == "maddpg":
            agent_n = [MADDPG(args, agent_id) for agent_id in range(args.N)]
            for agent_id in range(args.N):
                agent_n[agent_id].load_model(os.path.join(load_nn, 'actor_{}_{}k.pth'.format(agent_id, args.iteration)))
            test_model(env, agent_n=agent_n, render=args.render, max_episode_steps=args.max_episode_steps)
        elif args.policy == "matd3":
            agent_n = [MATD3(args, agent_id) for agent_id in range(args.N)]
            for agent_id in range(args.N):
                agent_n[agent_id].load_model(os.path.join(load_nn, 'actor_{}_{}k.pth'.format(agent_id, args.iteration)))
            test_model(env, agent_n=agent_n, render=args.render, max_episode_steps=args.max_episode_steps)
        else:
            print(f"{args.policy} is unsupported policy")
            sys.exit()