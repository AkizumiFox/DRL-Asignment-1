import math
import random
from collections import deque

def get_sign(n):
    if n > 0: return 1
    if n < 0: return -1
    return 0

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

class SimpleTaxiEnv():
    def __init__(self, grid_size=10, fuel_limit=5000):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit

        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        
        self.obstacles = None
        self.stations = None
        self.passenger_loc = None
        self.destination = None

        self.taxi_pos = None

    def generate_obstacles(self):
        """
        Generate floor(0.1 * grid_size^2) obstacles and ensure all blank cells are connected.
        Will keep regenerating obstacles until a valid configuration is found.
        """
        num_obstacles = math.floor(0.1 * self.grid_size * self.grid_size)
        
        while True:
            # Consider all grid positions for obstacles.
            all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
            obstacles = set(random.sample(all_positions, num_obstacles))
            
            # List of blank (non-obstacle) positions.
            blank_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                          if (x, y) not in obstacles]
            
            # Use BFS to check connectivity among blank cells.
            start = blank_cells[0]
            visited = set()
            queue = deque([start])
            while queue:
                cell = queue.popleft()
                if cell in visited:
                    continue
                visited.add(cell)
                x, y = cell
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and
                        (nx, ny) not in obstacles and (nx, ny) not in visited):
                        queue.append((nx, ny))
            
            if len(visited) == len(blank_cells):
                self.obstacles = obstacles
                break

    def generate_stations(self):
        """
        Generate 4 random grid positions as stations (from blank cells)
        ensuring no two are adjacent (i.e. they are not connected by an edge).
        """
        # Stations must be chosen only from blank cells.
        blank_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                       if (x, y) not in self.obstacles]
        random.shuffle(blank_cells)
        stations = []
        for pos in blank_cells:
            valid = True
            for s in stations:
                # Manhattan distance of 1 means adjacent.
                if abs(pos[0] - s[0]) + abs(pos[1] - s[1]) == 1:
                    valid = False
                    break
            if valid:
                stations.append(pos)
            if len(stations) == 4:
                break
        if len(stations) < 4:
            raise ValueError("Not enough blank cells to place 4 non-adjacent stations.")
        self.stations = stations

    def generate_locs(self):
        lst = random.sample([0, 1, 2, 3], 2)
        self.passenger_loc = self.stations[lst[0]]
        self.destination = self.stations[lst[1]]

    def generate_taxi_pos(self):
        available_positions = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if (x, y) not in self.stations and (x, y) not in self.obstacles
        ]
        self.taxi_pos = random.choice(available_positions)

    def reset(self):
        """
        Reset the environment:
          - Generate obstacles first.
          - Then generate stations from the blank cells.
          - Place the taxi in a blank cell that is not a station.
          - Randomly choose a passenger location from the stations.
          - Choose a destination from the remaining stations.
        """
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        
        self.generate_obstacles()
        self.generate_stations()
        self.generate_locs()
        self.generate_taxi_pos()
        
        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        
        if action == 0:  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        
        if action in [0, 1, 2, 3]:
            if ((next_row, next_col) in self.obstacles or
                not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size)):
                reward -= 5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos  
                else:
                    reward = -10  
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward - 0.1, True, {}
                    else:
                        reward -= 10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -= 10
                    
        reward -= 0.1  
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward - 10, True, {}
        
        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination
        
        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = (passenger_loc_north or passenger_loc_south or 
                          passenger_loc_east or passenger_loc_west or passenger_loc_middle)
       
        destination_loc_north = int((taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int((taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int((taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int((taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle = int((taxi_row, taxi_col) == self.destination)
        destination_look = (destination_loc_north or destination_loc_south or 
                            destination_loc_east or destination_loc_west or destination_loc_middle)
        
        state = (taxi_row, taxi_col,
                 self.stations[0][0], self.stations[0][1],
                 self.stations[1][0], self.stations[1][1],
                 self.stations[2][0], self.stations[2][1],
                 self.stations[3][0], self.stations[3][1],
                 obstacle_north, obstacle_south, obstacle_east, obstacle_west,
                 passenger_look, destination_look)
        return state

    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        """Print the environment grid and current status to the terminal."""
        # Create a grid filled with dots.
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        # Mark the station positions with numbers 1 to 4.
        for i, station in enumerate(self.stations, start=1):
            x, y = station
            grid[x][y] = str(i)
        
        # Mark obstacles.
        for ox, oy in self.obstacles:
            grid[ox][oy] = 'X'
        
        # Place taxi.
        ty, tx = taxi_pos
        if 0 <= ty < self.grid_size and 0 <= tx < self.grid_size:
            grid[ty][tx] = 'T'

        # Print step information and fuel.
        # print(f"\nStep: {step}")
        # print(f"Taxi Position: ({tx}, {ty})")
        # print(f"Fuel Left: {fuel}")
        # print(f"Last Action: {self.get_action_name(action)}\n")

        # Print the grid.
        for row in grid:
            print(" ".join(str(cell) for cell in row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None and 0 <= action < len(actions) else "None"

def interactive_session():
    """Run an interactive session in the terminal."""
    env = SimpleTaxiEnv(grid_size=10, fuel_limit=5000)
    state, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    prev_reward = 0
    last_action = None
    passenger_is_on_taxi = 0  # Initialize passenger indicator to 0 (not on taxi)
    
    print("Welcome to the SimpleTaxiEnv interactive session!")
    print("Actions: 0=Move South, 1=Move North, 2=Move East, 3=Move West, 4=Pick Up, 5=Drop Off, q=Quit")
    
    while not done:
        env.render_env(env.taxi_pos, action=last_action, step=step_count, fuel=env.current_fuel)
        
        # Print the raw state observation
        print("\nRaw State Observation:")
        print(state)
        print(f"Passenger on taxi: {'Yes' if passenger_is_on_taxi else 'No'}")
        print(f"Sign Distance: {get_sign_distance(state)}")
        
        user_input = input("Enter action (or 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Exiting interactive session.")
            break
        
        try:
            action = int(user_input)
            if action < 0 or action > 5:
                raise ValueError
        except ValueError:
            print("Invalid action. Please enter a number between 0 and 5 or 'q' to quit.")
            continue
        
        prev_state = state
        state, reward, done, _ = env.step(action)
        # Update passenger_is_on_taxi using the passenger_on_taxi function
        passenger_is_on_taxi = passenger_on_taxi(prev_state, action, state, passenger_is_on_taxi)
        
        prev_reward = reward
        total_reward += reward
        last_action = action  # Store the action for the next iteration
        print(f"Reward: {reward}")
        step_count += 1
        
    print("Session ended.")
    print(f"Final Total Reward: {total_reward}")

if __name__ == "__main__":
    interactive_session()
