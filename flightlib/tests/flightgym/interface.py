import os
import numpy as np

import flightgym
from flightgym import QuadrotorEnv_v1
from flightgym import TargetTrackingEnv_v0


def main():
  quad_env1 = QuadrotorEnv_v1()
  obs = np.zeros(shape=(100, 12), dtype=np.float32)
  quad_env1.reset(obs)
  print(obs)

  target_env1 = TargetTrackingEnv_v0()
  obs = np.zeros(shape=(100, 12), dtype=np.float32)
  target_obs = np.zeros(shape=(12,), dtype=np.float32)
  target_env1.reset(obs, target_obs)
  print(obs)
  print(target_obs)

  a = np.zeros([4], dtype=np.float32) # target ground truth state
  print(a.shape)

if __name__ == "__main__":
    main()
