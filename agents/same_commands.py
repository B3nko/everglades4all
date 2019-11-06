# Standard library imports
import os
import time
import pdb

# Specialized imports
import numpy as np

NODE_CONNECTIONS = {
    1: [2, 4],
    2: [1, 3, 5],
    3: [2, 4, 5, 6, 7],
    4: [1, 3, 7],
    5: [2, 3, 8, 9],
    6: [3, 9],
    7: [3, 4, 9, 10],
    8: [5, 9, 11],
    9: [5, 6, 7, 8, 10],
    10: [7, 9, 11],
    11: [8, 10]
}


NUM_GROUPS = 12

ENV_MAP = {
    'everglades': 'Everglades-v0',
    'everglades-vision': 'EvergladesVision-v0',
    'everglades-stoch': 'EvergladesStochastic-v0',
    'everglades-vision-stoch': 'EvergladesVisionStochastic-v0',
}

class same_commands:
    def __init__(self, action_space, player_num):
        self.action_space = action_space
        self.num_groups = NUM_GROUPS

        self.num_nodes = len(NODE_CONNECTIONS)
        self.num_actions = action_space

        self.shape = (self.num_actions, 2)

        self.first_turn = True
        self.steps = 0
        self.player_num = player_num
        #print('player_num: {}'.format(player_num))

        self.groups = []

    # end __init__

    def get_action(self, obs):
        #print('!!!!!!! Observation !!!!!!!!')
        ##print(obs)
        #print(obs[0])
        #for i in range(45,101,5):
        #    print(obs[i:i+5])
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        action = np.zeros(self.shape)

        # The next line should really be 0, but there is an env bug that the agent doesn't get
        # to see the 0th observation to make it's first move; one turn gets blown
        if not self.first_turn:
            action = self.act_test_turn(action)
        else:
            action = self.act_test_turn(action)
            self.first_turn = False
        #print(action)
        
        return action
    # end get_action

    def act_test_turn(self,actions):
        for i in range(1,8):
            actions[i-1] = [i, i]
        return actions

# end class
