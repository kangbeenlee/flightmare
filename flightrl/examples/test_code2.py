import time
import numpy as np
import os


from ruamel.yaml import YAML, dump, RoundTripDumper
# from rpg_baselines.envs import env_wrapper as wrapper
from rpg_baselines.envs import vec_env_wrapper as wrapper
from flightgym import QuadrotorEnv_v1


def main():

    # cfg_path = os.path.abspath("../configs/env.yaml")
    # cfg = YAML().load(open(cfg_path, 'r'))

    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))

    # env = DynamicGate_v0(dump(cfg["env"], Dumper=RoundTripDumper))
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))

    render = True

    obs = env.reset()
    print(obs.shape)

    if render:
        env.connectUnity()
        
    for i in range(10000):
        act = np.array([10.0, 0.0, 0.0, 0.0])
        next_obs, rew, done, _ = env.step(act)
        time.sleep(0.01)
        
    if render:
        env.disconnectUnity()

if __name__ == "__main__":
    main()
