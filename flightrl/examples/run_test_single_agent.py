#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

import os
import sys
import numpy as np
import torch
import random

from rpg_baselines.envs import target_tracking_env_wrapper as wrapper
from flightgym import TargetTrackingEnv_v0



def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_model(env, render=False):
    num_rollouts = 20
    
    if render:
        env.connectUnity()

    for n_roll in range(num_rollouts):
        obs = env.reset()
        target_obs = env.get_target_state()

        done, ep_len = False, 0

        while not done:
            ep_len += 1

            # vx, vy, vz, wz (m/s, m/s, m/s, rad/s)
            act = np.array([[0.0, 0.0, 0.0, 2.0]], dtype=np.float32)
            
            # # Step input response test
            # vx = 3.0
            # vy = -3.0
            # vz = 0.0
            # wz = 1.0
            
            # if ep_len < 150:
            #     act = np.array([[vx, vy, vz, wz]], dtype=np.float32)
            # elif 150 <= ep_len < 300:
            #     act = np.array([[-vx, -vy, -vz, -wz]], dtype=np.float32)
            # elif 300 <= ep_len < 450:
            #     act = np.array([[vx, vy, vz, wz]], dtype=np.float32)
            # elif 450 <= ep_len < 600:
            #     act = np.array([[-vx, -vy, -vz, -wz]], dtype=np.float32)
            # else:
            #     act = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            
            obs, rew, done, infos = env.step(act)
            target_obs = env.get_target_state()

    if render:
        env.disconnectUnity()

def main():
    # Environment setting parameter
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/target_tracking_env.yaml", 'r'))
    cfg["env"]["num_envs"] = 1
    cfg["env"]["num_threads"] = 1
    cfg["env"]["scene_id"] = 0
    cfg["env"]["num_targets"] = 4
    cfg["env"]["render"] = "yes"

    # Environment and policy type information
    print("Test environment name : Flightrl Single Agent Reinforcement Learning Environment")
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

    # Start single test!
    test_model(env, render=True)
        
if __name__ == "__main__":
    main()