# File containing global parameters for the project
import numpy as np

# --- ENVIRONMENT SETTINGS ---
X_LIM = (-15, 15)
Y_LIM = (-10, 10)

# --- ROBOT SETTINGS ---
SENSING_RANGE = 3.0
ROBOT_RADIUS = 0.5
OBS_INFLATION_RADIUS = 0.5 # should be = ROBOT_RADIUS to ensure safety
DIST_BTW_ROBOTS = 0.1 # Minimum distance to maintain between robots to avoid collisions
BL_RANGE = SENSING_RANGE * 0.65  # radius of the sensing range for the treasure detecting

NUM_ROBOTS = 3
TX_INTERVAL_TICKS = NUM_ROBOTS * 1 # Number of simulation steps between each broadcast opportunity for the robots

# --- MAPPING SETTINGS ---
MAP_RESOLUTION =ROBOT_RADIUS * 0.8 # Distance between points in the generated map

# --- RAY CASTING SETTINGS ---
# NB if this value is too small, the simulation will lag a bit
RAY_CASTING_RESOLUTION = 10.0 # degrees between rays 

# --- FORMATION CONTROL SETTINGS ---
RING_RADIUS = BL_RANGE * 0.8 # Distance from the goal at which the robot is considered

# --- EKF SETTINGS ---
# NB: These matrices contains the std devs not the variances
SIGMA_V = 0.05 # m/s
SIGMA_W = 0.01 # rad/s
EKF_PROCESS_NOISE_STD = np.array([SIGMA_V, SIGMA_W])     # Used for the process noise in the EKF (Cov matrix Q)
CAM_NOISE_STD = np.array([0.1, 0.1, 0.1])    # USed for the measurement noise in the camera (Cov matrix R)

# --- LIDAR NOISE ---
LIDAR_NOISE_STD = 0.05 # Standard deviation of the noise added to the LiDAR measurements (in meters) 



