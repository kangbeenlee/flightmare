import matplotlib.pyplot as plt
import numpy as np
import copy



def test_model(env, agent_n=None, render=False, max_episode_steps=500):
    num_rollouts = 100
    
    if render:
        env.connectUnity()
        
    for n_roll in range(num_rollouts):
        episode_reward = 0
        obs_n, done_n, epi_step = env.reset(), False, 0
        while epi_step < max_episode_steps:
            epi_step += 1
            # We do not add noise when evaluating
            a_n = np.array([agent.choose_action(obs, noise_std=0, train=False) for agent, obs in zip(agent_n, obs_n)]).astype(np.float32)
            obs_n, r_n, done_n, _ = env.step(copy.deepcopy(a_n))
            episode_reward += np.sum(r_n)

            if all(done_n):
                break

        print(">>> Evaluation episode {}, reward: {:.1f}".format(n_roll, episode_reward))

    if render:
        env.disconnectUnity()