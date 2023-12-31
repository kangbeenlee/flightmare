import matplotlib.pyplot as plt
import numpy as np


def test_model(env, model=None, render=False, max_episode_steps=500):
    num_rollouts = 1001

    if render:
        env.connectUnity()
        
    for n_roll in range(num_rollouts):
        score = 0.0
        obs, done, epi_step = env.reset(), False, 0
        while not (done or (epi_step > max_episode_steps)):
            epi_step += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            score += reward[0]

        print(">>> Evaluation episode {}, reward: {:.1f}".format(n_roll, score))

    if render:
        env.disconnectUnity()