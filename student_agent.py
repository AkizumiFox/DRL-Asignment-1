import numpy as np 
import pickle
import random
import gym
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

def q_table_fac():
    return np.zeros(6)

def get_station_directions(obs):
    """
    Computes the relative directions for each station with respect to the taxi.
    Returns 16 binary values:
      - For each station: [is east, is west, is south, is north]
    """
    taxi_row, taxi_col = obs[0], obs[1]
    stations = [
        (obs[2], obs[3]),
        (obs[4], obs[5]),
        (obs[6], obs[7]),
        (obs[8], obs[9])
    ]
    station_directions = []
    for station in stations:
        station_row, station_col = station
        station_directions.append(int(station_col > taxi_col))
        station_directions.append(int(station_col < taxi_col))
        station_directions.append(int(station_row > taxi_row))
        station_directions.append(int(station_row < taxi_row))
    return tuple(station_directions)

def passenger_on_taxi(prev_state, action, now_state, prev_flag):
    """
    Determines if the passenger is picked up.
    Returns 1 if passenger is picked up (and 0 otherwise).
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

    if action == 5 and [taxi_row_2, taxi_col_2] in stations_2 and destination_look_2 == 1:
        return 0
    if action != 4:
        return prev_flag
    if [taxi_row_2, taxi_col_2] in stations_2 and passenger_look_2 == 1:
        return 1
    return 0

def update_target(obs, old_target):
    """
    Updates the target station index based on the taxi's position.
    """
    if (obs[0], obs[1]) == (obs[2], obs[3]) and old_target == 0:
        return 1
    if (obs[0], obs[1]) == (obs[4], obs[5]) and old_target == 1:
        return 2
    if (obs[0], obs[1]) == (obs[6], obs[7]) and old_target == 2:
        return 3
    if (obs[0], obs[1]) == (obs[8], obs[9]) and old_target == 3:
        return 0
    return old_target

def reward_shaping(prev_obs, prev_target, action, now_obs, now_target, reward):
    """
    Applies potential-based reward shaping based on distance to current target.
    """
    def get_target_coords(obs, target_idx):
        if target_idx == 0: return obs[2], obs[3]
        elif target_idx == 1: return obs[4], obs[5]
        elif target_idx == 2: return obs[6], obs[7]
        elif target_idx == 3: return obs[8], obs[9]

    def manhattan_distance(row1, col1, row2, col2):
        return abs(row1 - row2) + abs(col1 - col2)

    if prev_target != now_target:
        return reward
    else:
        target_row, target_col = get_target_coords(now_obs, now_target)
        prev_potential = -manhattan_distance(prev_obs[0], prev_obs[1], target_row, target_col)
        next_potential = -manhattan_distance(now_obs[0], now_obs[1], target_row, target_col)
        shaping_reward = next_potential - prev_potential
        return reward + shaping_reward

# ---------------------------
# DRQN Network and Agent
# ---------------------------
class DQRNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, lstm_layers=1):
        super(DQRNNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        # Initial fully-connected layer to process the state
        self.fc1 = nn.Linear(state_size, hidden_size)
        # LSTM layer to capture temporal dependencies
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        # Final layer to output Q-values for each action
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x, hidden_state):
        """
        Forward pass with explicit hidden state management
        Args:
            x: Input tensor of shape (batch_size, state_size)
            hidden_state: Tuple (h_0, c_0) of LSTM hidden states
        Returns:
            q_values: Q-values for each action
            next_hidden: Updated hidden state
        """
        # Ensure x has a sequence dimension (batch, seq_len, feature)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # shape: (batch_size, 1, state_size)
        
        x = torch.relu(self.fc1(x))  # shape: (batch_size, seq_len, hidden_size)
        
        # Pass through LSTM with explicit hidden state
        x, next_hidden = self.lstm(x, hidden_state)
        
        # Use the output from the last time step
        x = x[:, -1, :]
        q_values = self.fc2(x)
        
        return q_values, next_hidden
    
    def reset_hidden_state(self, batch_size=1):
        """Creates a new hidden state tensor filled with zeros"""
        device = next(self.parameters()).device
        h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

class DRQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lstm_layers=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        # Create Q-network for policy
        self.policy_net = DQRNNetwork(state_size, action_size, hidden_size, lstm_layers).to(self.device)
        self.epsilon = 0.01  # Small exploration probability for evaluation

    def reset_hidden_state(self):
        """Reset the hidden state for a new episode"""
        return self.policy_net.reset_hidden_state()

    def act(self, state, hidden_state):
        """
        Choose an action using an epsilon-greedy policy.
        state: tuple or numpy array of shape (state_size,)
        hidden_state: current hidden state tuple for the LSTM.
        Returns: action (int) and updated hidden state.
        """
        # Expand dimensions: (1, 1, state_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            # Update hidden state with a forward pass
            with torch.no_grad():
                _, hidden_state = self.policy_net(state_tensor, hidden_state)
            return action, hidden_state
        else:
            self.policy_net.eval()
            with torch.no_grad():
                q_values, hidden_state = self.policy_net(state_tensor, hidden_state)
            self.policy_net.train()
            action = torch.argmax(q_values, dim=1).item()
            return action, hidden_state

    def load(self, filename):
        """Load model parameters from file"""
        checkpoint = torch.load(filename, map_location="cpu")
        self.policy_net.load_state_dict(checkpoint['policy_model_state_dict'])
        print(f"Loaded model from {filename}")

def get_action(obs):
    """
    Main function called by the evaluation environment.
    Manages the agent and returns the selected action.
    """
    STATE_SIZE = 24
    ACTION_SIZE = 6 

    # Initialize agent and state attributes on first call
    if not hasattr(get_action, "agent"):
        get_action.agent = DRQNAgent(STATE_SIZE, ACTION_SIZE)
        get_action.agent.load("dqn_checkpoint_ep80600.pt")
        get_action.hidden_state = get_action.agent.reset_hidden_state()
        get_action.have_passenger = 0
        get_action.now_target = 0
        get_action.prev_obs = obs
    else:
        # Update passenger status and target based on previous action
        prev_action = getattr(get_action, "prev_action", 0)
        get_action.have_passenger = passenger_on_taxi(
            get_action.prev_obs, prev_action, obs, get_action.have_passenger
        )
        get_action.now_target = update_target(obs, get_action.now_target)
    
    # Construct state vector - same format as used during training
    station_directions = get_station_directions(obs)
    state = np.array(
        list(station_directions) + 
        list(obs[10:]) + 
        [get_action.have_passenger] + 
        [get_action.now_target],
        dtype=np.float32
    )
    
    # Get action from agent, passing current hidden state
    action, get_action.hidden_state = get_action.agent.act(state, get_action.hidden_state)
    action_map = {
        0: 3, 
        1: 2, 
        2: 0, 
        3: 1, 
        4: 4, 
        5: 5
    }
    action = action_map[action]
    
    # Store for next call
    get_action.prev_obs = obs
    get_action.prev_action = action
    
    return action
