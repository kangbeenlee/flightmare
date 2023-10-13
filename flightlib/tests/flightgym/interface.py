import os
import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper

import flightgym
from flightgym import TargetTrackingEnv_v0


def main():
  cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/target_tracking_env.yaml", 'r'))
  target_env1 = TargetTrackingEnv_v0(dump(cfg, Dumper=RoundTripDumper), False)
  obs = np.zeros(shape=(100, 12), dtype=np.float32)
  target_obs = np.zeros(shape=(12,), dtype=np.float32)
  target_env1.reset(obs, target_obs)
  print(obs)
  print(target_obs)

if __name__ == "__main__":
    main()
