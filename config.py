# File containing global parameters for the project


# --- ENVIRONMENT SETTINGS ---
X_LIM = (-15, 15)
Y_LIM = (-10, 10)

# --- ROBOT SETTINGS ---
SENSING_RANGE = 3.0
ROBOT_RADIUS = 0.5
OBS_INFLATION_RADIUS = 0.5 # should be = ROBOT_RADIUS to ensure safety
DIST_BTW_ROBOTS = 0.1 # Minimum distance to maintain between robots to avoid collisions
BL_RANGE = SENSING_RANGE * 0.8  # radius of the sensing range for the treasure detecting

NUM_ROBOTS = 3
TX_INTERVAL_TICKS = NUM_ROBOTS * 1 # Number of simulation steps between each broadcast opportunity for the robots

# --- MAPPING SETTINGS ---
MAP_RESOLUTION =ROBOT_RADIUS * 0.5 # Distance between points in the generated map

# --- RAY CASTING SETTINGS ---
# NB if this value is too small, the simulation will lag a bit
RAY_CASTING_RESOLUTION = 10.0 # degrees between rays 

# --- FORMATION CONTROL SETTINGS ---
RING_RADIUS = BL_RANGE * 0.8 # Distance from the goal at which the robot is considered


