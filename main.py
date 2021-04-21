import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent
import sys
sys.path.append('/usr/local/lib/python3.9/site-packages')
import gym

plt.xlabel('Episode')
plt.ylabel('Reward')
env = gym.make('FrozenLake-v0')

if __name__ == '__main__':
    win_pct = []
    scores = []
    n_episodes = 1000000
    agent = Agent(lr=0.001, gamma=0.99, eps=1.0, eps_min=0.0001, eps_decay=0.9999995,
                  num_states=16, num_actions=4)
    for ep in range(n_episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, observation_)
            observation = observation_
            score += reward
        scores.append(score)
        if ep % 100 == 0:
            score_mean = np.mean(scores)
            win_pct.append(score_mean)
            score = 0
            scores = []
            if ep % 1000 == 0:
                print('episode:', ep, 'win_pct: %.2f' % score_mean, 'epsilon: %.2f' % agent.eps)

env.close()
plt.plot(win_pct)
plt.show()
