import numpy as np
from scipy.stats import levy_stable

class LevyController:
    """
    External controller for Lévy Flight movement.
    Maintains its own internal state to avoid polluting the Robot class.
    """
    def __init__(self, alpha=1.5, t_min=1.0):
        # Distribution parameters
        self.alpha = alpha
        self.t_min = t_min
        
        # Internal navigation state
        self.time_remaining = 0.0
        self.target_theta = 0.0

    def get_control(self, robot_state, dt):
        """
        Calculates the [v, w] commands based on the current robot state.
        
        Args:
            robot_state: The numpy array [x, y, theta] representing the robot's state
            dt: Time step
            
        Returns:
            A numpy array containing the linear and angular velocities [v, w]
        """
        x, y, theta = robot_state
        
        # 1. If the available time is over, generate a new "mission"
        if self.time_remaining <= 0:
            self.target_theta = np.random.uniform(-np.pi, np.pi)
            raw_time = levy_stable.rvs(alpha=self.alpha, beta=1.0, loc=self.t_min, scale=1.0)
            self.time_remaining = min(abs(raw_time), 20.0) 
            
        # 2. Angular error calculation
        angle_error = self.target_theta - theta
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi 

        # 3. Implicit State Machine: Rotate -> Move Forward
        if abs(angle_error) > 0.05: 
            v = 0.0
            w = 2.0 * angle_error # Rotate in place
        else:
            v = 1.0 # Move forward
            w = 0.0
            self.time_remaining -= dt # Consume the flight time

        return np.array([v, w])