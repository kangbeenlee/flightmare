import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


def test_model(env, render=False):
    num_rollouts = 20
    
    if render:
        env.connectUnity()

    for n_roll in range(num_rollouts):
        obs = env.reset()
        target_obs = env.get_target_state()
        ep_len = 0

        while True:
            ep_len += 1

            # vx, vy, vz, wz (m/s, m/s, m/s, rad/s)
            act = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            
            # Step input response test
            vx = 3.0
            vy = -3.0
            vz = 0.0
            wz = 1.0
            
            if ((ep_len // 150) % 2 == 0): # 3secs
                act = np.array([[-vx, -vy, -vz, -wz],
                                [vx, vy, vz, wz],
                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            else:
                act = np.array([[vx, vy, vz, wz],
                                [-vx, -vy, -vz, -wz],
                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            
            obs_n, r_n, done_n, _ = env.step(act)
            target_obs = env.get_target_state()

            if all(done_n):
                break

    if render:
        env.disconnectUnity()