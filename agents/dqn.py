import os
import time
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-5               # learning rate
UPDATE_EVERY = 4        # how often to update the network

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dqn:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, player_num, seed):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def get_action(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Updates state for the Q network
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Is local evalutation giving us state evaluation?
        self.qnetwork_local.eval()

        # Not sure what no_grad() is doing (no gradiant?)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)  # get action values?

        # Does this train the network? if we only play one game I am sure the network does not train well
        # need to figure out how to actually train a network
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:  # exploit
            # return np.argmax(action_values.cpu().data.numpy())
            # create the empty action space return value
            actions = np.zeros((7, 2))

            # not sure what np.flip does, maybe converts action values to a usable format and
            # gets the best actions (what does cpu() do?)
            prioritized_actions = np.flip(action_values.cpu().data.numpy().argsort())
            #print(f"prioritized actions = {prioritized_actions}\n\n")
            selected_groups = []

            for action in prioritized_actions[0]:
                # get the group from the action
                group = np.floor(action / 11.).astype(int)

                # get the node from the action
                node = int(action % 11) + 1

                # if we haven't tried to move the group yet (we can only move a group once)
                # add the group movement to the array of actions
                if group not in selected_groups:
                    actions[len(selected_groups), 0] = group
                    actions[len(selected_groups), 1] = node
                    selected_groups.append(group)

                # we can only move 7 groups
                if len(selected_groups) >= 7:
                    break

            return actions
        else:  # explore (choose a random option)
            # return random.choice(np.arange(self.action_size))
            actions = np.zeros((7, 2))
            actions[:, 0] = np.random.choice(12, 7, replace=False)
            actions[:, 1] = np.random.choice(11, 7, replace=False) + 1
            return actions

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            where:
                s = current state
                a = action
                r = reward
                s' = new state
                done = ?

            gamma (float): discount factor
        """
        print(f'learning...')
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # look up detach
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        print(f'loss = {loss}')

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def watch_untrained_agent(env, agent):
    state = env.reset()
    for step in range(200):
        actions = np.zeros((7, 2))
        groups_to_move = np.random.choice(12, 7, replace=False)
        for i, group in enumerate(groups_to_move):
            state[0] = group  # replace step number with group
            action = agent.act(state)
            actions[i, 0] = group
            actions[i, 1] = action
        state, reward, done, info = env.step(actions)
        if done:
            break
    return


def train_dqn(env, agent, n_episodes=2000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.999):
    """
    Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores_deque = deque(maxlen=100)   # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            actions = agent.act(state, eps)
            # print(actions)

            next_state, reward, done, _ = env.step(actions)

            # DQN step() can only train one action at a time, so step 7 times
            for index in range(actions.shape[0]):
                top_action = int(actions[index, 0] *
                                 11 + actions[index, 1] - 1)
                agent.step(state, top_action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break

        scores_deque.append(score)       # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('Episode {}\tAverage Score: {:.4f}\tEpisode Score: {:.4f}'.format(
            i_episode, np.mean(scores_deque), score))

        if i_episode > 100 and np.mean(scores_deque) >= 0.8:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_deque)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return


def main(pub_socket=None):
    server_address = os.getenv('SERVER_ADDRESS', 'server')
    pub_socket = int(os.getenv('PUB_SOCKET', pub_socket))
    if pub_socket is None:
        raise Exception('Pub socket not set')

    # print(f'Pub socket is {pub_socket}')

    env_config = {
        'await_connection_time': 120,
        'server_address':  server_address,
        'pub_socket': pub_socket,
        'sub_socket': '5563',
    }

    _env_name = os.getenv('ENV_NAME', 'everglades')

    render_image = os.getenv('RENDER_IMAGE', 'false').lower() == 'true'
    viewer = None

    env_name = ENV_MAP[_env_name.lower()]
    env = gym.make(env_name, env_config=env_config)

    # DQN picks the highest value action from the available actions
    # To make this feasible, each group-action combination must be an output
    agent = Agent(state_size=105, action_size=12*11, seed=0)

    # watch_untrained_agent(env, agent)
    train_dqn(env, agent)


if __name__ == "__main__":
    main()
