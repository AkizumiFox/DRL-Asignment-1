# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque 

# Neural Network for DQN
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = deque(maxlen=buffer_size)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)

        # Convert to numpy arrays first for better performance
        observations = np.array(observations, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_observations = np.array(next_observations, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return (
            torch.tensor(observations).to(self.device),
            torch.tensor(actions).to(self.device),
            torch.tensor(rewards).to(self.device),
            torch.tensor(next_observations).to(self.device),
            torch.tensor(dones).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)

# DQN Agent updated for once-per-episode learning and epsilon decay per episode
class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=64,
                 gamma=0.99, learning_rate=0.001, tau=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.tau = tau      # for soft update of target network

        # Q-Networks (policy and target)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999  # Decay applied once per episode

    def act(self, obs, eval_mode=False):
        """Select an action using epsilon-greedy policy"""
        if (not eval_mode) and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(obs)
        self.policy_net.train()

        return torch.argmax(action_values, dim=1).item()

    def step(self, obs, action, reward, next_obs, done):
        """Add experience to memory.
           Epsilon decay is now handled once per episode in the training loop."""
        self.memory.add(obs, action, reward, next_obs, done)

    def learn(self):
        """Perform a single learning step using a batch of experience tuples"""
        # Sample from memory
        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)

        # Get Q values for current states
        Q_expected = self.policy_net(observations).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max Q values for next states from target model
        with torch.no_grad():
            Q_targets_next = self.target_net(next_observations).max(1)[0]

        # Compute target Q values
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Calculate loss
        loss = nn.MSELoss()(Q_expected, Q_targets)

        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # Soft update target network
        self.soft_update()

    def soft_update(self):
        """Soft update of target network parameters"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        """Save the model"""
        torch.save({
            'policy_model_state_dict': self.policy_net.state_dict(),
            'target_model_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        """Load the model"""
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['policy_model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

def get_action(obs):
    """
    Given an observation, returns the action chosen by the trained agent.
    If the observation doesn't match the expected shape, a fallback random action is used.
    """
    STATE_SIZE = 16
    ACTION_SIZE = 6 

    # Initialize the agent only once
    if not hasattr(get_action, "agent"):
        get_action.agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
        get_action.agent.load("dqn_checkpoint_ep100000.pt")

    # Ensure obs is a NumPy array with dtype float32
    obs = np.array(obs, dtype=np.float32)
    if obs.shape[0] != STATE_SIZE:
        print("Size error!")
        return random.choice(range(ACTION_SIZE))
    return get_action.agent.act(obs)