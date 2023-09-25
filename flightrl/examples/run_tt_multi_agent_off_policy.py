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
from rpg_baselines.multi_agent.off_policy.replay_buffer import ReplayBuffer
from rpg_baselines.multi_agent.off_policy.maddpg import MADDPG
from rpg_baselines.multi_agent.off_policy.matd3 import MATD3
# from rpg_baselines.multi_agent.test import test_model
from rpg_baselines.multi_agent.test_control import test_model


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.render = args.render        
        # Create env
        self.env = env

        # Create N agents
        if self.args.policy == "maddpg":
            self.agent_n = [MADDPG(args, agent_id) for agent_id in range(args.N)]
        elif self.args.policy == "matd3":
            self.agent_n = [MATD3(args, agent_id) for agent_id in range(args.N)]
        else:
            print(f"{self.args.policy} is not supported marl policy")
            print("--------------------------------------------------------------------------------------------")

        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/{}'.format(self.args.policy))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        
        self.time_steps = 0 # Total training time step
        self.tqdm_bar = tqdm(initial=0, desc="Training", total=self.args.max_training_timesteps, unit="timestep")

        self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def run(self, ):
        if self.render:
            self.env.connectUnity()

        self.evaluate_policy()

        while self.time_steps < self.args.max_training_timesteps:
            obs_n = self.env.reset()
            
            for _ in range(self.args.max_episode_steps):
                self.tqdm_bar.update(1)
                
                # Each agent selects actions based on its own local observations(add noise for exploration)
                a_n = np.array([agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, obs_n)]).astype(np.float32)
                # --------------------------!!!! 주의!!!여기서 deepcopy, MPE 환경은 a_n에 5를 곱합니다.-------------------------------------------
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
                    for agent_id in range(self.args.N):
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
                a_n = np.array([agent.choose_action(obs, noise_std=0) for agent, obs in zip(self.agent_n, obs_n)]).astype(np.float32)  # We do not add noise when evaluating
                obs_next_n, r_n, done_n, _ = self.env.step(copy.deepcopy(a_n))
                episode_reward += np.sum(r_n)
                obs_n = obs_next_n
                if all(done_n):
                    break
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluation_times
        self.evaluate_rewards.append(evaluate_reward)
        print("time_steps:{} \t\t evaluate_reward:{} \t\t noise_std:{}".format(self.time_steps, evaluate_reward, self.noise_std))
        self.writer.add_scalar("evaluate_step_rewards", evaluate_reward, global_step=self.time_steps)
        # Save the rewards and models
        if not os.path.exists("./marl_train_data"):
            os.makedirs("./marl_train_data")
        np.save("./marl_train_data/{}_.npy".format(self.args.policy), np.array(self.evaluate_rewards))
        for agent_id in range(self.args.N):
            self.agent_n[agent_id].save_model(os.path.join("./", "model_multi"), self.args.policy, self.time_steps, agent_id)


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3")

    parser.add_argument('--n', type=int, default=4, help="Number of agent")
    parser.add_argument('--train', action="store_true", help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=1, help="Enable Unity Render")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--load_nn', type=str, default='./model_multi', help='Trained actor weight path for ddpg and td3')
    
    parser.add_argument("--max_training_timesteps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--max_episode_steps", type=int, default=500, help="Maximum number of steps per episode")
    parser.add_argument("--evaluation_time_steps", type=float, default=5000, help="Evaluate the policy every 'evaluation_time_steps'")
    parser.add_argument("--evaluation_times", type=float, default=5, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=3.0, help="Max action")

    parser.add_argument("--policy", type=str, default="maddpg", help="maddpg or matd3")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5, help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate of critic")
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
        args.device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(args.device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")

    # Environment setting parameter
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/target_tracking_env.yaml", 'r'))
    if args.train:
        cfg["env"]["num_envs"] = args.n
        cfg["env"]["scene_id"] = 0
    else:
        cfg["env"]["num_envs"] = args.n
        cfg["env"]["num_threads"] = 1
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
    args.N = env.num_envs  # The number of agents
    args.obs_dim_n = [env.num_obs for i in range(args.N)]  # obs dimensions of N agents
    args.action_dim_n = [env.num_acts for i in range(args.N)]  # actions dimensions of N agents

    print("--------------------------------------------------------------------------------------------")
    print("Max training timesteps :", args.max_training_timesteps)
    print("Max timesteps per episode :", args.max_episode_steps)
    print("Evaluation timesteps :", args.evaluation_time_steps)
    print("--------------------------------------------------------------------------------------------")
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
        if args.policy == "maddpg":
            agent_n = [MADDPG(args, agent_id) for agent_id in range(args.N)]
            # for i in range(args.N):
            #     agent_n[i].load_model(os.path.join(args.load_nn, f"maddpg_actor_step_910k_agent_{i}.pth"))
            # test_model(env, agent_n=agent_n, render=args.render, max_episode_steps=args.max_episode_steps)
            test_model(env, render=args.render)
        elif args.policy == "matd3":
            agent_n = [MATD3(args, agent_id) for agent_id in range(args.N)]
            for i in range(args.N):
                agent_n[i].load_model(os.path.join(args.load_nn, f"matd3_actor_step_790k_agent_{i}.pth"))
            test_model(env, agent_n=agent_n, render=args.render, max_episode_steps=args.max_episode_steps)
            # test_model(env, render=args.render)
        else:
            print(f"{args.policy} is unsupported policy")
            print("--------------------------------------------------------------------------------------------")
            sys.exit()