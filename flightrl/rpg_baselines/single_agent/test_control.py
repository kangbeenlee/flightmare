import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


def test_model(env, render=False):
    num_rollouts = 5
    
    if render:
        env.connectUnity()

    for n_roll in range(num_rollouts):
        obs = env.reset()
        target_obs = env.get_target_state()

        done, ep_len = False, 0

        while not done:

            # # vx, vy, vz, wz (m/s, m/s, m/s, rad/s)
            act = np.array([[0.0, 0.2, 0.0, 0.5]], dtype=np.float32)
            
            # # Step input response test
            # vx = 0.0
            # vy = 1.0
            # vz = 0.0
            # wz = 0.0
            
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
            #
            ep_len += 1

    if render:
        env.disconnectUnity()