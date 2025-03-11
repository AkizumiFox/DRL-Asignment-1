import numpy as np 
import pickle
import random
import gym
import math
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.tau = tau      # for soft update of target network

        # Q-Networks (policy and target)
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
        self.epsilon_decay = 0.99995  # Decay applied once per episode

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

# Function to preprocess the state for the neural network
def preprocess_state(state):
    """Convert tuple state to numpy array for neural network input"""
    return np.array(state, dtype=np.float32)

def get_distances(state):
    """Computes Manhattan distances from the taxi to each station given the raw state."""
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    stations = [[0, 0] for _ in range(4)]
    (
        taxi_row, taxi_col,
        stations[0][0], stations[0][1],
        stations[1][0], stations[1][1],
        stations[2][0], stations[2][1],
        stations[3][0], stations[3][1],
        _, _, _, _,
        _, _
    ) = state
    distances = [manhattan_distance((taxi_row, taxi_col), station) for station in stations]
    return distances

def passenger_on_taxi(prev_state, action, now_state, prev):
    """
    Determines whether the passenger is on the taxi.
    Returns 1 if the passenger is on board, 0 otherwise.
    """
    stations_1 = [[0, 0] for _ in range(4)]
    (
        taxi_row_1, taxi_col_1,
        stations_1[3][0], stations_1[3][1],
        stations_1[2][0], stations_1[2][1],
        stations_1[1][0], stations_1[1][1],
        stations_1[0][0], stations_1[0][1],
        _, _, _, _,
        passenger_look_1, destination_look_1
    ) = prev_state

    stations_2 = [[0, 0] for _ in range(4)]
    (
        taxi_row_2, taxi_col_2,
        stations_2[3][0], stations_2[3][1],
        stations_2[2][0], stations_2[2][1],
        stations_2[1][0], stations_2[1][1],
        stations_2[0][0], stations_2[0][1],
        _, _, _, _,
        passenger_look_2, destination_look_2
    ) = now_state

    # If a drop-off action occurs and the taxi is at a station with destination visible, flag resets.
    if action == 5 and [taxi_row_2, taxi_col_2] in stations_2 and destination_look_2 == 1:
        return 0
    # For actions other than pickup, carry forward previous flag.
    if action != 4:
        return prev
    # If pickup action and taxi is at a station with passenger visible, set flag.
    if [taxi_row_2, taxi_col_2] in stations_2 and passenger_look_2 == 1:
        return 1
    return 0

def get_action(obs):
    STATE_SIZE = 21
    ACTION_SIZE = 6 

    if not hasattr(get_action, "agent"):
        get_action.agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
        get_action.agent.load("dqn_checkpoint_ep45200.pt")
    
    if not hasattr(get_action, "prev_obs"):
        get_action.have_passenger = 0
    else:
        get_action.have_passenger = passenger_on_taxi(get_action.prev_raw_obs, get_action.prev_action, obs, get_action.have_passenger)

    get_action.prev_raw_obs = obs   
    obs = (obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], obs[9], obs[13], obs[12], obs[11], obs[10], obs[14], obs[15])
    distances = get_distances(obs)[::-1]
    
    this_obs = list(obs) + list(distances) + [get_action.have_passenger]
    this_obs = np.array(this_obs, dtype=np.float32)

    action = get_action.agent.act(this_obs)
    get_action.prev_action = action

    return action