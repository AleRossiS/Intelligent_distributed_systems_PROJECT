import numpy as np
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import Intelligent_distributed_systems_PROJECT.config as cfg


## HYPERPARAMETERS ##
MAX_TIME = 10       # Maximum time to keep moving in the same direction
MIN_TIME = 2        # Minimum time to keep moving in the same direction
LIN_VEL = 1.0       # Linear velocity magnitude
ANG_VEL_GAIN = 1.5  # Angular velocity gain for turning
ALPHA = 1.5         # Lévy distribution parameter (0 < alpha <= 2)
BETA = 1.0          # Lévy distribution skewness parameter (-1 <= beta <= 1)
SCALE = 1.0         # Lévy distribution scale parameter (scale > 0
        
        

class MotionController:
    """
    External controller that generates the control inputs for the robots [v, w].
    
    This class implements:
        - A Lévy Flight-based control strategy for exploration and treasure hunt.
        -
        -
    """
    def __init__(self):
        ## Lévy Distribution parameters ##

        # Internal navigation state
        self.time_remaining = 0.0
        self.target_theta = 0.0
        self.avoidance_dir = None # direction in whitch to rotate when near boundary (1 for CCW, -1 for CW)

    def levy_control(self, robot_state, dt, voronoi, centroid):
        """
        Calculates the [v, w] commands based on Levy Flight using the current robot state.
        If the robot is too close to its Voronoi boundary, it will turn towards the interior 
        of its cell and sample a new flight time. Otherwise, it will keep moving forward until 
        the flight time runs out.
        
        Args:
            robot_state: The numpy array [x, y, theta] representing the robot's state
            dt: Time step
            
        Returns:
            A numpy array containing the linear and angular velocities controls [v, w] 
        """

        x, y, theta = robot_state # Current orientation of the robot
        robot_pos = Point(x, y) # Current position as a Point object
        
        # 1. Create a line in the direction of the robot's heading and check distance to Voronoi boundary
        heading_line = LineString([(x, y), (x + cfg.SENSING_RANGE*1.2 * np.cos(theta), y + cfg.SENSING_RANGE*1.2 * np.sin(theta))])

        is_near_boundary = False # Initialize the flag
        
        if voronoi.exterior.intersects(heading_line):
            intersection = voronoi.exterior.intersection(heading_line) # Find the intersection point
            
        # Compute distance to the boundary
        if not intersection.is_empty:
            if intersection.geom_type == 'Point':
                dist = robot_pos.distance(intersection)
                
            # Check if the robot is within a certain threshold distance to the boundary
            if dist < cfg.SENSING_RANGE * 0.5:
                is_near_boundary = True
            
        # 2. If near the boundary, set a new target angle towards the centroid and sample a new flight time
        if is_near_boundary:
            
            if self.avoidance_dir is None:
                # Decide randomly whether to turn clockwise or counterclockwise
                self.avoidance_dir = np.random.choice([-1, 1])
            
            self.target_theta = theta + np.pi/8*(self.avoidance_dir)# New Angle towards the centroid
            
            # Sample from Lévy distribution new time
            raw_time = levy_stable.rvs(alpha=ALPHA,    # stability parameter (0 < alpha <= 2)
                                       beta=BETA,      # skewness parameter (-1 <= beta <= 1)
                                       loc=MIN_TIME,   # location parameter
                                       scale=SCALE)    # scale parameter (scale > 0)
            
            self.time_remaining = np.clip(abs(raw_time), MIN_TIME, MAX_TIME)  # Ensure time is within bounds
        else:
            self.avoidance_dir = None # Reset the boundary avoidance direction when not near boundary anymore
        
        # 3. If not near boundary just do Levy Flight until time runs out
        if self.time_remaining <= 0:
            self.target_theta = np.random.uniform(-np.pi, np.pi) # Random new direction
            
            #Sample from Levy dsitrib
            raw_time = levy_stable.rvs(alpha=ALPHA,  # stability parameter (0 < alpha <= 2)
                                       beta=BETA,     # skewness parameter (-1 <= beta <= 1)
                                       loc=MIN_TIME, # location parameter
                                       scale=SCALE)    # scale parameter (scale > 0)
            
            self.time_remaining = np.clip(abs(raw_time), MIN_TIME, MAX_TIME)  # Ensure time is within bounds
            
        # 4. Angular error calculation
        angle_error = self.target_theta - theta
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi # Normalize to [-pi, pi]

        # 5. Implicit State Machine: Rotate -> Move Forward
        if abs(angle_error) > 0.05: 
            v = 0.0
            w = ANG_VEL_GAIN * angle_error # Rotate in place
        else:
            v = LIN_VEL # Move forward
            w = 0.0
            self.time_remaining -= dt # Decrease the remaining time 

        return np.array([v, w])
    
    
    def plot_levy_distribution(self, num_samples=10000):
        """
        Plots the Lévy distribution for visualization and debugging purposes.
        """
        
        x = np.linspace(0, 15, num_samples)
        pdf = levy_stable.pdf(x, alpha=ALPHA, beta=BETA, loc=MIN_TIME, scale=SCALE)

        plt.ioff() 
        
        fig = plt.figure(figsize=(8, 5))
        plt.plot(x, pdf, color='blue', linewidth=2)
        
        plt.axvline(x=MIN_TIME, color='red', linestyle='--', linewidth=2, label=f'MIN_TIME ({MIN_TIME})')
        plt.axvline(x=MAX_TIME, color='green', linestyle='--', linewidth=2, label=f'MAX_TIME ({MAX_TIME})')
        
        plt.title("Levy Distribution PDF (alpha={}, beta={})".format(ALPHA, BETA))
        plt.xlabel("Time Duration")
        plt.ylabel("Probability Density")
        plt.grid(True)
        
        plt.legend()
        plt.show()
        

            