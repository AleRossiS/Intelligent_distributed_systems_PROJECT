# File containing global parameters for the project


# --- ENVIRONMENT SETTINGS ---
X_LIM = (-10, 10)
Y_LIM = (-10, 10)

# --- ROBOT SETTINGS ---
SENSING_RANGE = 3.0
ROBOT_RADIUS = 0.5
OBS_INFLATION_RADIUS = 0.5 # should be = ROBOT_RADIUS to ensure safety
DIST_BTW_ROBOTS = 0.1 # Minimum distance to maintain between robots to avoid collisions
BL_RANGE = SENSING_RANGE * 0.8  # radius of the sensing range for the treasure detecting


# --- RAY CASTING SETTINGS ---
# NB if this vaule is too small, the simulation will lag a bit
RAY_CASTING_RESOLUTION = 10.0 # degrees between rays 
