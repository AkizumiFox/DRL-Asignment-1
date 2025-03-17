import numpy as np 
import pickle
import random
import gym
import math
from collections import deque

def get_sign(n):
    if n > 0: return 1
    if n < 0: return -1
    return 0

def passenger_on_taxi(prev_state, action, now_state, prev):
    stations_1 = [[0, 0] for _ in range(4)]
    (
        taxi_row_1, taxi_col_1,
        stations_1[0][0], stations_1[0][1],
        stations_1[1][0], stations_1[1][1],
        stations_1[2][0], stations_1[2][1],
        stations_1[3][0], stations_1[3][1],
        _, _, _, _,
        passenger_look_1, destination_look_1
    ) = prev_state

    stations_2 = [[0, 0] for _ in range(4)]
    (
        taxi_row_2, taxi_col_2,
        stations_2[0][0], stations_2[0][1],
        stations_2[1][0], stations_2[1][1],
        stations_2[2][0], stations_2[2][1],
        stations_2[3][0], stations_2[3][1],
        _, _, _, _,
        passenger_look_2, destination_look_2
    ) = now_state

    if prev == 1:
        return 1
    if action == 4 and [taxi_row_2, taxi_col_2] in stations_2 and passenger_look_2 == 1:
        return 1
    return 0

def get_distance(obs):
    taxi_row, taxi_col = obs[0], obs[1]
    stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]

    lst = []
    for (station_row, station_col) in stations:
        lst.append((station_row - taxi_row, station_col - taxi_col))
    return lst

def get_sign_distance(obs):
    dis = get_distance(obs)
    lst = []
    for (row, col) in dis:
        lst.append((get_sign(row), get_sign(col)))
    return lst

def now_on_station(obs):
    taxi_row, taxi_col = obs[0], obs[1]
    stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]

    for (station_row, station_col) in stations:
        if taxi_row == station_row and taxi_col == station_col:
            return True
    return False

def get_agent_state(obs, have_passenger, vis):
    """
    Convert the observation and passenger flag into the agent's state representation.
    - get_sign_distance(obs) returns a list of 4 tuples; we convert it to a tuple of tuples.
    - obs[10:16] are the six remaining observation features.
    - have_passenger is the passenger flag.
    """
    # sign_distance = list(tuple(pair) for pair in get_sign_distance(obs))
    features = tuple(obs[10:16])
    vistied = [(obs[0] + dx, obs[1] + dy) in vis for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    vistied = tuple(vistied)
    on_station = now_on_station(obs)
    return (features, vistied, have_passenger, on_station)


def get_action(obs):
    if not hasattr(get_action, "q_table"):
        get_action.q_table = pickle.load(open("q_table6.pkl", "rb"))
        get_action.have_passenger = 0
        get_action.vis = {(obs[0], obs[1])}
    else:
        get_action.have_passenger = passenger_on_taxi(get_action.prev_obs, get_action.prev_action, obs, get_action.have_passenger)
        get_action.vis.add((obs[0], obs[1]))

    state = get_agent_state(obs, get_action.have_passenger, get_action.vis)

    if state not in get_action.q_table or np.random.uniform(0, 1) < 0.01: this_action = np.random.randint(0, 5)
    else: this_action = np.argmax(get_action.q_table[state])

    get_action.prev_obs = obs
    get_action.prev_action = this_action
    
    return this_action