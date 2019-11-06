import os
import numpy as np
import time

class random_actions:
    def __init__(self, action_space, player_num):
        self.action_space = action_space
        self.num_groups = 12

        self.num_nodes = 11
        self.num_actions = action_space

        self.shape = (self.num_actions, 2)

    def get_action(self, obs):
        #print('!!!!!!! Observation !!!!!!!!')
        #print(obs)
        #print(obs[0])
        #for i in range(45,101,5):
        #    print(obs[i:i+5])
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        action = np.zeros(self.shape)
        action[:, 0] = np.random.choice(self.num_groups, self.num_actions, replace=False)
        action[:, 1] = np.random.choice(np.arange(1, self.num_nodes + 1), self.num_actions, replace=False)
        #print(action)
        return action


