import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork, DuelingQNetwork

BUFFER_SIZE = int(1e5)              # replay buffer size
BATCH_SIZE = 64                     # minibatch size
GAMMA = 0.99                        # discount factor
TAU = 1e-3                          # for soft update of target parameters
LR = 1e-3                           # learning rate
UPDATE_EVERY = 4                    # how often to update the network
DOUBLE_DQN = True                   # use double dqn
DUELING_DQN = True                 # use dueling dqn
PRIORITIZED_EXPERIENCE = False      # use prioritized experience replay
PRIORITIZED_EPSILON = 1e-2          # epsilon for experience priorities so to make all probabilities non-zero
PRIORITIZED_POWER = 0.1             # power to raise priorities to 0 = uniform distribution

HIDDEN_SIZES = [64,64]              # list of sizes for hidden layers
DUELING_SIZES = [64,64]                # list of sizes for hidden layers for dueling streams (only used if DUELING_DQN == TRUE)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
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
        if DUELING_DQN:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, HIDDEN_SIZES, DUELING_SIZES, seed).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, HIDDEN_SIZES, DUELING_SIZES, seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, HIDDEN_SIZES, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, HIDDEN_SIZES, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)


        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, beta):
        # Save experience in replay memory
        priority = None
        if PRIORITIZED_EXPERIENCE:
            priority = self.get_priority(state, action, reward, next_state, done)
        self.memory.add(state, action, reward, next_state, done, priority)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, beta)
                
    def get_priority(self, state, action, reward, next_state, done):
        """Returns the the absolute error of your QNetwork to be used in 
        prioritized experience replay

        Params
        =====
          state (array_like): state where we decided to take action
          action (array_like): action taken in state
          reward (float): reward received after taking action in state
          next_state (array_like): current state after we took action in state
          done (bool): episode is finished

        """
        state = torch.from_numpy(state.reshape(1,-1)).float().to(device)
        next_state = torch.from_numpy(next_state.reshape(1,-1)).float().to(device)
        if DOUBLE_DQN:
            Q_actions_next = self.qnetwork_local(next_state).detach().argmax(1)[0]
            Q_targets_next = self.qnetwork_target(next_state).detach()[0][Q_actions_next].item()
        else:
            Q_targets_next = self.qnetwork_target(next_state).detach().max(1)[0].item()
        # Compute Q targets for current states 
        Q_target = reward + (GAMMA * Q_targets_next * (1 - done))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(state).detach()[0][action].item()
        return abs(Q_target - Q_expected)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, beta):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights = experiences

        # Get max predicted Q values (for next states) from target model
        if DOUBLE_DQN:
            Q_actions_next = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_actions_next)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = torch.sum(weights.pow(beta) * (Q_expected - Q_targets).pow(2))
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
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
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        if PRIORITIZED_EXPERIENCE:
            self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, priority=None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        if PRIORITIZED_EXPERIENCE:
            self.priorities.append((priority + PRIORITIZED_EPSILON) ** PRIORITIZED_POWER)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if PRIORITIZED_EXPERIENCE:
            prob = np.array(self.priorities)/np.sum(self.priorities)
            experience_idxs = np.random.choice(len(self.memory), size=self.batch_size, p=prob)
            experiences = [self.memory[i] for i in experience_idxs]
            #calculate importance-sampling weight to correct for bias
            weight = torch.from_numpy(np.vstack([1./(len(prob)*p) for e, p in zip(experiences, prob) if e is not None])).float().to(device)
        else:
            experiences = random.sample(self.memory, k=self.batch_size)
            weight = torch.from_numpy(np.vstack([1 for e in experiences if e is not None])).float().to(device)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

  
        return (states, actions, rewards, next_states, dones, weight)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
