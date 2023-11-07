import matplotlib.pyplot as plt
import numpy as np


# def test_model(env, model=None, render=False, max_episode_steps=500):
#     num_rollouts = 601

#     if render:
#         env.connectUnity()
        
#     for n_roll in range(num_rollouts):
#         score = 0.0
#         obs, done, epi_step = env.reset(), False, 0
        
#         while not (done or (epi_step > max_episode_steps)):
#             # print(obs[0, 3:6])
#             epi_step += 1
#             action = model.select_action(obs)
  
#             obs, reward, done, infos = env.step(action)
#             score += reward[0]

#         print(">>> Evaluation episode {}, reward: {:.1f}".format(n_roll, score))

#     if render:
#         env.disconnectUnity()


def test_model(env, model=None, render=False, max_episode_steps=500):
    num_rollouts = 1001
    main = 2

    if render:
        env.connectUnity()
        
    for n_roll in range(num_rollouts):
        score = 0.0
        obs, done, epi_step = env.reset(), False, 0
        obs = obs[main]

        while not (done or (epi_step > max_episode_steps)):
            epi_step += 1
            action = model.select_action(np.array(obs)).reshape(1, -1).astype(np.float32)
            pseudo_action = np.array([[0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            pseudo_action = np.concatenate((pseudo_action, action), axis=0)

            obs, reward, done, _ = env.step(pseudo_action)
            obs = obs[main]
            done = done[main]

            score += reward[main]

        print(">>> Evaluation episode {}, reward: {:.1f}".format(n_roll, score))

    if render:
        env.disconnectUnity()