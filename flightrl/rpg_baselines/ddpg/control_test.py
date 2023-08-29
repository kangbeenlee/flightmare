import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


def test_model(env, render=False):
    max_ep_length = env.max_episode_steps
    print(">>> max_ep_length:", max_ep_length)
    
    num_rollouts = 5
    
    if render:
        env.connectUnity()

    for n_roll in range(num_rollouts):
        pos, euler, dpos, deuler = [], [], [], []
        actions = []
        obs = env.reset()
        target_obs = env.get_target_state()

        done, ep_len = False, 0

        # while not (done or (ep_len >= max_ep_length)):
        while not done:

            # # vx, vy, vz, wz (m/s, m/s, m/s, rad/s)
            # act = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            
            # Step input response test
            vx = 0.0
            vy = 0.0
            vz = 0.0
            wz = 0.0
            
            print("obs: ", obs)
            print("target obs: ", target_obs)
            
            if ep_len < 150:
                act = np.array([[vx, vy, vz, wz]], dtype=np.float32)
            elif 150 <= ep_len < 300:
                act = np.array([[-vx, -vy, -vz, -wz]], dtype=np.float32)
            elif 300 <= ep_len < 450:
                act = np.array([[vx, vy, vz, wz]], dtype=np.float32)
            elif 450 <= ep_len < 600:
                act = np.array([[-vx, -vy, -vz, -wz]], dtype=np.float32)
            else:
                act = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            
            
            obs, rew, done, infos = env.step(act)
            target_obs = env.get_target_state()
            #
            ep_len += 1
            #
            pos.append(obs[0, 0:3].tolist())
            dpos.append(obs[0, 6:9].tolist())
            euler.append(obs[0, 3:6].tolist())
            deuler.append(obs[0, 9:12].tolist())
            #
            actions.append(act[0, :].tolist())
            
        pos = np.asarray(pos)
        dpos = np.asarray(dpos)
        euler = np.asarray(euler)
        deuler = np.asarray(deuler)
        actions = np.asarray(actions)

    if render:
        env.disconnectUnity()