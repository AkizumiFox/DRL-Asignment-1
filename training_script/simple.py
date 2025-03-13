import gym
import numpy as np
import time
from IPython.display import clear_output

class SimpleTaxiEnv(gym.Wrapper):
    def __init__(self, fuel_limit=5000):
        self.grid_size = 5
        env = gym.make("Taxi-v3", render_mode="ansi")  # Taxi-v3 is always 5x5
        super().__init__(env)

        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit

        # Four corner stations in a 5x5 grid
        self.stations = [(0, 0), (0, self.grid_size - 1),
                         (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None
        self.passenger_picked_up = False  
        self.obstacles = set()  # No obstacles in simple version
        self.destination = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.current_fuel = self.fuel_limit

        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(obs)

        # The “stations” correspond to the possible passenger/dest positions in vanilla Taxi
        self.passenger_loc = self.stations[pass_idx]
        self.destination = self.stations[dest_idx]
        self.passenger_picked_up = False  

        return self.get_state(), info

    def get_state(self):
        """
        Construct a 'raw state' tuple. The example below includes:
          (taxi_row, taxi_col, station_1_x, station_1_y, ..., obstacle_north, obstacle_south, ...
           passenger_look, destination_look)
        """
        taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(self.env.unwrapped.s)
        obstacle_north = int(taxi_row == 0)
        obstacle_south = int(taxi_row == self.grid_size - 1)
        obstacle_east  = int(taxi_col == self.grid_size - 1)
        obstacle_west  = int(taxi_col == 0)
        
        # “Look” flags – if the passenger or destination is near or exactly where the taxi is
        passenger_look = int((taxi_row, taxi_col) == self.passenger_loc)
        destination_look = int((taxi_row, taxi_col) == self.destination)

        state = (
            taxi_row, taxi_col,
            self.stations[0][0], self.stations[0][1],
            self.stations[1][0], self.stations[1][1],
            self.stations[2][0], self.stations[2][1],
            self.stations[3][0], self.stations[3][1],
            obstacle_north, obstacle_south,
            obstacle_east, obstacle_west,
            passenger_look,
            destination_look
        )
        return state

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(self.env.unwrapped.s)

        # Attempt move
        next_row, next_col = taxi_row, taxi_col
        if action == 0:  # Move South
            next_row += 1
        elif action == 1:  # Move North
            next_row -= 1
        elif action == 2:  # Move East
            next_col += 1
        elif action == 3:  # Move West
            next_col -= 1

        # If we tried to move out of bounds, penalize, but remain in the same spot
        if action in [0, 1, 2, 3]:
            if not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward = -5
                self.current_fuel -= 1
                done = self.current_fuel <= 0
                return self.get_state(), (reward - 10 if done else reward), done, False, {}

        # Decrement fuel if we move in bounds
        self.current_fuel -= 1

        # Let the original Taxi-v3 environment handle transitions/logic
        obs, reward, terminated, truncated, info = super().step(action)

        # Adjust the default Taxi-v3 reward scheme to your preference:
        if reward == 20:
            reward = 50
        elif reward == -1:
            reward = -0.1
        elif reward == -10:
            reward = -10

        # If picking up
        if action == 4:
            # pass_idx=4 in Taxi-v3 means passenger is in the taxi
            if pass_idx == 4:
                self.passenger_picked_up = True
            else:
                self.passenger_picked_up = False

        # If dropping off
        elif action == 5:
            if self.passenger_picked_up:
                # If at the destination
                if (taxi_row, taxi_col) == self.destination:
                    reward += 50
                    # End the episode
                    return self.get_state(), reward, True, {}, {}
                else:
                    reward -= 10

        # Keep passenger location updated if they’re picked up
        if self.passenger_picked_up:
            self.passenger_loc = (taxi_row, taxi_col)

        # End episode if out of fuel
        if self.current_fuel <= 0:
            return self.get_state(), reward - 10, True, False, {}

        return self.get_state(), reward, False, truncated, info

    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        """
        Print out the environment in a “grid” style in the terminal.
        """
        # Build a textual 5x5 grid
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        # Mark the corner stations (red, green, yellow, blue)
        grid[0][0] = 'R'
        grid[0][self.grid_size - 1] = 'G'
        grid[self.grid_size - 1][0] = 'Y'
        grid[self.grid_size - 1][self.grid_size - 1] = 'B'

        ty, tx = taxi_pos
        if (0 <= ty < self.grid_size) and (0 <= tx < self.grid_size):
            grid[ty][tx] = '🚖'

        print(f"\nStep: {step}")
        print(f"Taxi Position: ({ty}, {tx})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


def run_interactive_game(fuel_limit=5000):
    """
    Run a loop letting the user choose actions at each step using W, A, S, D, P, O.
    """
    env = SimpleTaxiEnv(fuel_limit=fuel_limit)
    obs, info = env.reset()

    total_reward = 0.0
    done = False
    step_count = 0

    # Extract initial taxi position from the raw state (obs)
    # According to our get_state() => (taxi_row, taxi_col, ...)
    taxi_row, taxi_col = obs[0], obs[1]

    # A simple mapping from user keys to env actions
    #   0 = Move South, 1 = Move North, 2 = Move East, 3 = Move West, 4 = Pick Up, 5 = Drop Off
    key_to_action = {
        '0', '1', '2', '3', '4', '5'
    }

    while not done:
        # Render environment in the terminal
        env.render_env((taxi_row, taxi_col), action=None, step=step_count, fuel=env.current_fuel)

        # Show the raw state in the terminal
        print(f"Raw State: {obs}")

        # Prompt for user input
        action_str = input().lower()

        if action_str == 'q':
            print("Exiting the game.")
            break

        # Validate user input
        if action_str not in key_to_action:
            print("Invalid action. Use W, A, S, D, P, O (or Q to quit).")
            continue

        # action = key_to_action[action_str]
        action = int(action_str)

        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # Update taxi position from new obs
        taxi_row, taxi_col = obs[0], obs[1]

        # Print out the immediate reward
        print(f"Action: {env.get_action_name(action)}, Reward: {reward}")

        # If the environment says we're done, break
        if done:
            env.render_env((taxi_row, taxi_col), action=action, step=step_count, fuel=env.current_fuel)
            print(f"Episode finished after {step_count} steps!")
            print(f"Total Reward: {total_reward}")

    print("Game ended.")


if __name__ == "__main__":
    # Start the game in interactive mode with W, A, S, D, P, O controls
    run_interactive_game(fuel_limit=5000)
