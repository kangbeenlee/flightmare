from ruamel.yaml import YAML, dump, RoundTripDumper

import os
import sys
import math
import argparse
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from rpg_baselines.single_agent.common.policies import MlpPolicy, MlpLstmPolicy
from rpg_baselines.single_agent.ppo.ppo2 import PPO2
from rpg_baselines.single_agent.ppo.ppo2_test import test_model
import rpg_baselines.single_agent.common.util as U

from rpg_baselines.envs import target_tracking_env_wrapper as wrapper
from flightgym import TargetTrackingEnv_v0



def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=1, help="Enable Unity Render")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--load_nn', type=str, default='./model/ppo/ppo2.zip', help='Trained ppo2 weight path')
    return parser


def main():
    args = parser().parse_args()
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/target_tracking_env.yaml", 'r'))
    if args.train:
        cfg["env"]["num_envs"] = 10
        cfg["env"]["num_threads"] = 10
        cfg["env"]["scene_id"] = 0
    else:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["scene_id"] = 0
    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    # env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    env = wrapper.FlightmareTargetTrackingEnv(TargetTrackingEnv_v0(dump(cfg, Dumper=RoundTripDumper), False))

    # set random seed
    configure_random_seed(args.seed, env=env)

    print("--------------------------------------------------------------------------------------------")
    print("Observation space dimension :", env.num_obs)
    print("Action space dimension :", env.num_acts)
    print("--------------------------------------------------------------------------------------------")

    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))

        if args.render:
            env.connectUnity()

        log_dir = rsg_root + '/runs/single'
        save_dir = rsg_root + '/model/ppo'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, 'ppo2')

        # saver = U.ConfigurationSaver(log_dir=log_dir)
        model = PPO2(tensorboard_log=log_dir,
                     policy=MlpPolicy,
                     policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])], act_fun=tf.nn.relu),
                     env=env,
                     lam=0.95,
                     gamma=0.99, # lower 0.9 ~ 0.99
                     # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
                     n_steps=512,
                     ent_coef=0.00,
                     learning_rate=3e-4,
                     vf_coef=0.5,
                     max_grad_norm=0.5,
                     nminibatches=64,
                     noptepochs=10,
                     cliprange=0.2,
                     verbose=1)

        model.learn(total_timesteps=int(3e7),
                    # total_timesteps=int(25000000),
                    log_dir=save_dir,
                    logger=logger)
        model.save(save_dir)

        if args.render:
            env.disconnectUnity()

    # Testing mode with a trained weight
    else:
        model = PPO2.load(args.load_nn)
        test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()