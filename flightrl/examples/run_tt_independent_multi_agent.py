#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

import os
import sys
import numpy as np
import torch
import random
import argparse

from rpg_baselines.single_agent.ddpg import DDPG
from rpg_baselines.single_agent.td3 import TD3
from rpg_baselines.envs import target_tracking_env_wrapper as wrapper
from flightgym import TargetTrackingEnv_v0



def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_model(env, model, render=True):
    num_rollouts = 20
    max_episode_steps = 1001

    if render:
        env.connectUnity()

    for n_roll in range(num_rollouts):
        episode_reward = 0
        obs_n, done_n, epi_step = env.reset(), False, 0
        while epi_step < max_episode_steps:
            epi_step += 1
            # We do not add noise when evaluating
            a_n = model.select_action(obs_n)
            obs_n, r_n, done_n, _ = env.step(a_n)
            episode_reward += np.mean(r_n)

            if all(done_n):
                break

        print(">>> Evaluation episode {}, reward: {:.1f}".format(n_roll, episode_reward))

    if render:
        env.disconnectUnity()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2, help="Number of agent (tracker)")
    parser.add_argument('--n_targets', type=int, default=3, help="Number of target")
    parser.add_argument('--load_nn', type=str, default='./model/ddpg/actor.pkl', help='Trained actor weight path for ddpg, td3')
    parser.add_argument('--gpu_id', type=str, default='cuda:0', help='Choose gpu device id')
    parser.add_argument("--policy", type=str, default="ddpg", help='ddpg or td3')
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_z_score_normalization", type=bool, default=False, help="Z score normalization to observation")
    parser.add_argument("--max_action", type=float, default=3.0, help="Maximum action of actor output")
    args = parser.parse_args()

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
    cfg["env"]["num_envs"] = args.n
    cfg["env"]["num_threads"] = 1
    cfg["env"]["scene_id"] = 0
    cfg["env"]["num_targets"] = args.n_targets
    cfg["env"]["render"] = "yes"

    # Environment and policy type information
    print("Test environment name : Flightrl Multi Agent Reinforcement Learning Environment")
    print("Scene ID :", cfg["env"]["scene_id"])
    print("The number of tragets         :", cfg["env"]["num_targets"])
    print("The number of tracker (agent) :", cfg["env"]["num_envs"])
    print("--------------------------------------------------------------------------------------------")

    # Generate target tracking environment
    env = wrapper.FlightmareTargetTrackingEnv(TargetTrackingEnv_v0(dump(cfg, Dumper=RoundTripDumper), False))
    configure_random_seed(0, env=env)
    print("Observation space dimension :", env.num_obs)
    print("Action space dimension :", env.num_acts)
    print("--------------------------------------------------------------------------------------------")

    if args.policy == "ddpg":
        model = DDPG(args, obs_dim=env.num_obs, action_dim=env.num_acts)
        model.load(args.load_nn)
    elif args.policy == "td3":
        model = TD3(args, obs_dim=env.num_obs, action_dim=env.num_acts)
        model.load(args.load_nn)
    test_model(env, model=model)
        
if __name__ == "__main__":
    main()