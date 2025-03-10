import random

class Maze:
    def __init__(self, n):
        """
        Initialize the maze of size (n+2) x (n+2). This includes a border of walls
        around an n x n interior. If n is even, also carves row n and column n
        so they're not fully blocked.
        """
        self.n = n
        self.size = n + 2
        self.maze = [['#' for _ in range(self.size)] for _ in range(self.size)]
        
        # Two sets for storing coordinates of walls and blanks
        self.wall_coords = set()
        self.blank_coords = set()
        
        # List to store the station coordinates
        self.stations = []
        
        # Lists to store taxi and passenger information
        self.taxi_location = []
        self.pass_idx = None  # Station index (0-3) for passenger location
        self.dest_idx = None  # Station index (0-3) for destination
        self.passenger_loc = None  # Actual coordinates of passenger location
        self.destination = None  # Actual coordinates of destination
        self.passenger_picked_up = False  # Track if passenger is in taxi
        self.fuel = 5000  # Initialize fuel

        # Generate the maze on initialization
        self._generate_maze()
        
        # Fill in the wall_coords and blank_coords sets based on the final maze
        self._populate_coordinate_sets()
        
        # Select the stations
        self._select_stations()
        
        # Select taxi location and passenger information
        self._select_taxi_and_passenger()

    def _generate_maze(self):
        """
        Internal method to carve out the maze using a DFS approach.
        """
        # 1) Mark interior carveable cells
        for row in range(1, self.n + 1):
            for col in range(1, self.n + 1):
                # Standard "odd cell" carve
                if row % 2 == 1 and col % 2 == 1:
                    self.maze[row][col] = ' '
                # If n is even, also open up the entire row n and col n
                elif self.n % 2 == 0:
                    if row == self.n or col == self.n:
                        self.maze[row][col] = ' '

        # 2) Perform DFS from the top-left carveable cell (1, 1)
        visited = set()
        
        def dfs(r, c):
            visited.add((r, c))
            for (nr, nc) in self._neighbors(r, c):
                if (nr, nc) not in visited:
                    # Carve a path between (r, c) and (nr, nc)
                    wall_r = (r + nr) // 2
                    wall_c = (c + nc) // 2
                    self.maze[wall_r][wall_c] = ' '
                    dfs(nr, nc)

        # Run DFS
        dfs(1, 1)

    def _neighbors(self, r, c):
        """
        Yield neighbor cells that are 2 steps away in the four
        cardinal directions, randomly shuffled.
        """
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Ensure the neighbor is valid and currently carved (i.e., ' ').
            if 1 <= nr <= self.n and 1 <= nc <= self.n and self.maze[nr][nc] == ' ':
                yield nr, nc

    def _populate_coordinate_sets(self):
        """
        Populate wall_coords and blank_coords based on the final
        layout in self.maze.
        """
        for r in range(self.size):
            for c in range(self.size):
                if self.maze[r][c] == '#':
                    self.wall_coords.add((r, c))
                else:
                    self.blank_coords.add((r, c))

    def _select_stations(self):
        """
        Randomly selects 4 blank cells to serve as stations.
        Stores their coordinates in self.stations as a list of [row, col] lists.
        """
        # Convert blank_coords to a list for random sampling
        blank_list = list(self.blank_coords)
        
        # Randomly select 4 coordinates (or fewer if there aren't enough blanks)
        num_stations = min(4, len(blank_list))
        selected_coords = random.sample(blank_list, num_stations)
        
        # Convert each tuple to a list and store in self.stations
        self.stations = [list(coord) for coord in selected_coords]

    def _select_taxi_and_passenger(self):
        """
        Randomly selects a blank cell for the taxi (not a station).
        Randomly selects two different stations for passenger location and destination.
        """
        # Get blank cells that are not stations
        station_coords = {tuple(station) for station in self.stations}
        available_coords = list(self.blank_coords - station_coords)
        
        # Select taxi location from available blank cells
        if available_coords:
            taxi_coord = random.choice(available_coords)
            self.taxi_location = list(taxi_coord)
        else:
            self.taxi_location = []
            
        # Select passenger location and destination from different stations
        if len(self.stations) >= 2:
            # Randomly select two different station indices
            station_indices = list(range(len(self.stations)))
            selected_indices = random.sample(station_indices, 2)
            
            self.pass_idx = selected_indices[0]
            self.dest_idx = selected_indices[1]
            
            # Store the actual coordinates
            self.passenger_loc = self.stations[self.pass_idx]
            self.destination = self.stations[self.dest_idx]
        else:
            # Fallback if there are fewer than 2 stations
            self.pass_idx = None
            self.dest_idx = None
            self.passenger_loc = None
            self.destination = None

    def step(self, action):
        """
        Execute an action and update the environment state.
        
        Args:
            action (int): 
                0: Move south
                1: Move north
                2: Move east
                3: Move west
                4: Pick up passenger
                5: Drop off passenger
                
        Returns:
            tuple: (observation, reward, done, False, False)
                - observation: Current state
                - reward: Reward for the action
                - done: Whether the episode is terminated
                - False: Placeholder
                - False: Placeholder
        """
        # Initialize reward (each step costs -0.1)
        reward = -0.1
        
        # Reduce fuel by 1
        self.fuel -= 1
        
        # Check if fuel is empty
        if self.fuel <= 0:
            return self.get_state(), reward - 10, True, False, False  # Apply fuel exhaustion penalty
        
        # Movement actions
        if 0 <= action <= 3:
            # Movement directions as (dr, dc) for [south, north, east, west]
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            dr, dc = directions[action]
            
            # Calculate new position
            new_r, new_c = self.taxi_location[0] + dr, self.taxi_location[1] + dc
            
            # Check if the new position is a wall
            if (new_r, new_c) in self.wall_coords:
                reward -= 5  # Penalty for hitting a wall
                return self.get_state(), reward, False, False, False
            
            # Update taxi location
            self.taxi_location = [new_r, new_c]
            
        # Pick up passenger
        elif action == 4:
            # Check if taxi is at passenger location and passenger isn't already picked up
            if not self.passenger_picked_up and self.taxi_location == self.passenger_loc:
                self.passenger_picked_up = True
            else:
                reward -= 10  # Penalty for incorrect pickup
            
        # Drop off passenger
        elif action == 5:
            # Check if taxi is at destination and passenger is in the taxi
            if self.passenger_picked_up and self.taxi_location == self.destination:
                self.passenger_picked_up = False
                reward += 50  # Reward for successful delivery
                
                # Randomize new passenger and destination
                self._randomize_passenger_and_destination()
            else:
                reward -= 10  # Penalty for incorrect dropoff
        
        # Return the observation, reward, done flag, and two placeholders
        return self.get_state(), reward, False, False, False

    def _randomize_passenger_and_destination(self):
        """
        Randomly selects new passenger and destination stations.
        The passenger and destination will be at different stations.
        """
        if len(self.stations) >= 2:
            # Randomly select two different station indices
            station_indices = list(range(len(self.stations)))
            selected_indices = random.sample(station_indices, 2)
            
            self.pass_idx = selected_indices[0]
            self.dest_idx = selected_indices[1]
            
            # Store the actual coordinates
            self.passenger_loc = self.stations[self.pass_idx]
            self.destination = self.stations[self.dest_idx]
            
            # Reset passenger status
            self.passenger_picked_up = False

    def __str__(self):
        """
        Allows printing the maze object directly with `print(maze)`.
        """
        return "\n".join("".join(row) for row in self.maze)

    def get_state(self):
        """
        Returns the current state representation as a tuple containing:
        - Taxi position
        - Station coordinates
        - Information about obstacles in cardinal directions
        - Information about passenger and destination visibility
        
        Returns:
            tuple: State representation
        """
        taxi_row, taxi_col = self.taxi_location
        
        # Check for obstacles in each direction
        obstacle_north = int((taxi_row - 1, taxi_col) in self.wall_coords)
        obstacle_south = int((taxi_row + 1, taxi_col) in self.wall_coords)
        obstacle_east = int((taxi_row, taxi_col + 1) in self.wall_coords)
        obstacle_west = int((taxi_row, taxi_col - 1) in self.wall_coords)
        
        # Pad stations list to ensure we have 4 stations
        padded_stations = self.stations + [[0, 0]] * (4 - len(self.stations))
        
        # Check for passenger visibility in each direction
        passenger_loc = self.passenger_loc
        passenger_loc_north = int(passenger_loc == [taxi_row - 1, taxi_col])
        passenger_loc_south = int(passenger_loc == [taxi_row + 1, taxi_col])
        passenger_loc_east = int(passenger_loc == [taxi_row, taxi_col + 1])
        passenger_loc_west = int(passenger_loc == [taxi_row, taxi_col - 1])
        passenger_loc_middle = int(passenger_loc == [taxi_row, taxi_col])
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
        
        # Check for destination visibility in each direction
        destination = self.destination
        destination_loc_north = int(destination == [taxi_row - 1, taxi_col])
        destination_loc_south = int(destination == [taxi_row + 1, taxi_col])
        destination_loc_east = int(destination == [taxi_row, taxi_col + 1])
        destination_loc_west = int(destination == [taxi_row, taxi_col - 1])
        destination_loc_middle = int(destination == [taxi_row, taxi_col])
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle
        
        # Construct the state tuple
        state = (
            taxi_row - 1, 
            taxi_col - 1, 
            padded_stations[0][0] - 1, padded_stations[0][1] - 1,
            padded_stations[1][0] - 1, padded_stations[1][1] - 1,
            padded_stations[2][0] - 1, padded_stations[2][1] - 1,
            padded_stations[3][0] - 1, padded_stations[3][1] - 1,
            obstacle_north, obstacle_south, obstacle_east, obstacle_west,
            passenger_look, destination_look
        )
        
        return state

def interactive_test(maze_size=6):
    """
    Interactive testing function that allows manual control of the taxi.
    
    Args:
        maze_size (int): Size of the maze to generate
    """
    # Initialize maze
    maze_obj = Maze(maze_size)
    done = False
    total_reward = 0
    
    # Display the initial state
    clear_screen()
    display_state(maze_obj)
    
    # Main interaction loop
    while not done:
        # Get action from user
        action = get_user_action()
        
        if action == -1:
            print("Exiting interactive test...")
            break
            
        # Execute the action
        obs, reward, done, _, _ = maze_obj.step(action)
        total_reward += reward
        
        # Display the updated state
        clear_screen()
        display_state(maze_obj)
        display_observation(obs)
        
        # Show action result
        print(f"Action: {action_to_string(action)}")
        print(f"Reward: {reward:.1f}")
        print(f"Total Reward: {total_reward:.1f}")
        
        if done:
            print("Game over! (Fuel depleted)")
            
        # Success message for pickup and dropoff
        if action == 4 and maze_obj.passenger_picked_up:
            print("Passenger picked up successfully!")
        elif action == 5 and not maze_obj.passenger_picked_up and reward > 0:
            print("Passenger delivered successfully!")
            print("New passenger and destination have been assigned!")

def clear_screen():
    """Clear the terminal screen."""
    print("\033[H\033[J", end="")  # ANSI escape sequence to clear screen

def action_to_string(action):
    """Convert action number to descriptive string with key mapping."""
    actions = {
        0: "Move South (S)",
        1: "Move North (W)", 
        2: "Move East (D)",
        3: "Move West (A)",
        4: "Pick up passenger (P)",
        5: "Drop off passenger (O)"
    }
    return actions.get(action, "Unknown")

def get_user_action():
    """Get action input from the user using keyboard-style controls."""
    print("\nAvailable actions:")
    print("W: Move North")
    print("A: Move West")
    print("S: Move South")
    print("D: Move East")
    print("P: Pick up passenger")
    print("O: Drop off passenger")
    print("Q: Quit")
    
    # Action mapping from keyboard controls to action numbers
    action_map = {
        'w': 1,  # North
        'a': 3,  # West
        's': 0,  # South
        'd': 2,  # East
        'p': 4,  # Pick up
        'o': 5,  # Drop off
    }
    
    while True:
        user_input = input("\nEnter action: ").strip().lower()
        
        if user_input == 'q':
            return -1
        elif user_input in action_map:
            return action_map[user_input]
        else:
            print("Invalid input. Please use W/A/S/D for movement, P to pick up, O to drop off, or Q to quit.")

def display_state(maze_obj):
    """Display the current state of the maze with taxi, passenger, and destination."""
    # Create a visual representation of the maze
    original_maze = [row[:] for row in maze_obj.maze]
    
    # Transpose the maze (swap rows and columns)
    visual_maze = [[original_maze[j][i] for j in range(maze_obj.size)] for i in range(maze_obj.size)]
    
    # Mark stations with numbers (note: coordinates are now swapped)
    for i, station in enumerate(maze_obj.stations):
        r, c = station
        visual_maze[c][r] = str(i)  # Notice the swap of r,c to c,r
    
    # Mark taxi position with 'T' (coordinates swapped)
    taxi_r, taxi_c = maze_obj.taxi_location
    visual_maze[taxi_c][taxi_r] = 'T'  # Swap r,c
    
    # If passenger is not picked up, mark their position with 'P'
    if not maze_obj.passenger_picked_up and maze_obj.passenger_loc:
        p_r, p_c = maze_obj.passenger_loc
        if [p_r, p_c] != maze_obj.taxi_location:  # Don't overwrite taxi
            visual_maze[p_c][p_r] = 'P'  # Swap r,c
    
    # Mark destination with 'D'
    if maze_obj.destination:
        d_r, d_c = maze_obj.destination
        if [d_r, d_c] != maze_obj.taxi_location:  # Don't overwrite taxi
            visual_maze[d_c][d_r] = 'D'  # Swap r,c
    
    # Print the maze
    print("\nCurrent Maze (Transposed):")
    print("-" * (maze_obj.size + 2))
    for row in visual_maze:
        print("|" + "".join(row) + "|")
    print("-" * (maze_obj.size + 2))
    
    # Print status information
    print(f"\nTaxi Location: {maze_obj.taxi_location}")
    print(f"Passenger Station: {maze_obj.pass_idx}")
    print(f"Destination Station: {maze_obj.dest_idx}")
    print(f"Passenger Picked Up: {'Yes' if maze_obj.passenger_picked_up else 'No'}")
    print(f"Fuel Remaining: {maze_obj.fuel}")

def display_observation(obs):
    """Display the observation tuple in a readable format."""
    if not obs:
        print("No observation available")
        return
        
    # Unpack the observation
    taxi_row, taxi_col = obs[0], obs[1]
    stations = [
        [obs[2], obs[3]],  # Station 0
        [obs[4], obs[5]],  # Station 1
        [obs[6], obs[7]],  # Station 2
        [obs[8], obs[9]]   # Station 3
    ]
    obstacle_north = obs[10]
    obstacle_south = obs[11]
    obstacle_east = obs[12]
    obstacle_west = obs[13]
    passenger_visible = obs[14]
    destination_visible = obs[15]
    
    # Print in a formatted way
    print("\nObservation Details:")
    print(f"  {obs}")
    directions = []
    if obstacle_north: directions.append("North")
    if obstacle_south: directions.append("South")
    if obstacle_east: directions.append("East")
    if obstacle_west: directions.append("West")
    print(", ".join(directions) if directions else "None")
    print(f"  Passenger visible: {'Yes' if passenger_visible else 'No'}")
    print(f"  Destination visible: {'Yes' if destination_visible else 'No'}")

if __name__ == "__main__":
    # Choose between automated test and interactive test
    print("1: Run automated test")
    print("2: Run interactive test")
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "2":
        # Ask for maze size
        try:
            size = int(input("Enter maze size (5-10 recommended): ").strip())
            interactive_test(size)
        except ValueError:
            print("Invalid input. Using default size 6.")
            interactive_test(6)
    else:
        # Original automated test
        test_n = 6
        maze_obj = Maze(test_n)
        print(f"\nMaze for n={test_n}:")
        print(maze_obj)
        
        print(f"Stations: {maze_obj.stations}")
        print(f"Taxi location: {maze_obj.taxi_location}")
        print(f"Passenger station index: {maze_obj.pass_idx}")
        print(f"Passenger location: {maze_obj.passenger_loc}")
        print(f"Destination station index: {maze_obj.dest_idx}")
        print(f"Destination: {maze_obj.destination}")
        print(f"Passenger picked up: {maze_obj.passenger_picked_up}")
        print(f"Fuel: {maze_obj.fuel}")
        
        # Get and display the state
        state = maze_obj.get_state()
        print(f"\nCurrent state: {state}")
        
        # Test some actions
        print("\nTesting actions:")
        
        # Try to move south
        obs, reward, done, _, _ = maze_obj.step(0)
        print(f"Move south: Reward = {reward}, Done = {done}")
        print(f"New taxi location: {maze_obj.taxi_location}")
        print(f"Fuel remaining: {maze_obj.fuel}")
        display_observation(obs)
        
        # Try to pick up passenger (will likely fail since taxi is not at passenger location)
        obs, reward, done, _, _ = maze_obj.step(4)
        print(f"Pick up passenger: Reward = {reward}, Done = {done}")
        print(f"Passenger picked up: {maze_obj.passenger_picked_up}")
        print(f"Fuel remaining: {maze_obj.fuel}")
        display_observation(obs)
        
        # Test fuel depletion
        print("\nTesting fuel depletion:")
        maze_obj.fuel = 1  # Set fuel to 1 to test depletion on next step
        obs, reward, done, _, _ = maze_obj.step(1)
        print(f"Action after fuel depletion: Reward = {reward}, Done = {done}")
        print(f"Fuel remaining: {maze_obj.fuel}")
        display_observation(obs)
