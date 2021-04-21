import numpy as np
import random


class Agent:
    def __init__(self, lr, gamma, eps, eps_min, eps_decay, num_states, num_actions):
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.num_states = num_states
        self.num_actions = num_actions

        self.Q = {}
        self.init_QTable()

    def init_QTable(self):
        for state in range(self.num_states):
            for action in range(self.num_actions):
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        if random.uniform(0, 1) < self.eps:
            return random.randrange(self.num_actions)
        else:
            actions = np.array([self.Q[state, a] for a in range(self.num_actions)])
            return np.argmax(actions)

    def decrement_epsilon(self):
        self.eps = self.eps * self.eps_decay
        return max(self.eps_min, self.eps)

    def learn(self, state, action, reward, next_state):
        actions = np.array([self.Q[next_state, a] for a in range(self.num_actions)])
        max_action = np.argmax(actions)
        self.Q[(state, action)] = (1 - self.lr) * self.Q[(state, action)] + self.lr * (
                    reward + self.gamma * self.Q[(next_state, max_action)])
        self.decrement_epsilon()
