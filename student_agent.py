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
        tuple: 16 binary values where:
        - Values 0-3: Station 0's [North, East, West, South] position relative to taxi
        - Values 4-7: Station 1's [North, East, West, South] position relative to taxi
        - Values 8-11: Station 2's [North, East, West, South] position relative to taxi
        - Values 12-15: Station 3's [North, East, West, South] position relative to taxi
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

def passenger_on_taxi(prev_state, action, now_state, prev):
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
        return prev
    if [taxi_row_2, taxi_col_2] in stations_2 and passenger_look_2 == 1:
        return 1
    return 0

def update_target(obs, old_target):
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

def get_action(obs):
    # Initialize attributes on first call
    if not hasattr(get_action, "q_table"):
        get_action.q_table = pickle.load(open("q_table.pkl", "rb"))
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
    
    # Process observation to create state representation
    station_directions = get_station_directions(obs)
    state = tuple(
        list(station_directions) + 
        list(obs[10:14])[::-1] +
        list(obs[14:]) + 
        [get_action.have_passenger] + 
        [get_action.now_target]
    )
    
    # Use Q-table to select the best action - no exploration during inference
    if state not in get_action.q_table:
        action = random.randint(0, 5)
    else:
        action = np.argmax(get_action.q_table[state])
    
    # Store current observation and selected action for next call
    get_action.prev_obs = obs
    get_action.prev_action = action
    
    return action
