import numpy as np
from shapely.geometry import Point, LineString
import config as cfg


class GlobalCamera:
    """
    Simulates a global camera in the environment subject to occlusions (ray-casting)
    and Gaussian noise.
    """
    def __init__(self, x, y):
        self.pos = Point(x, y)
        # Standard deviations for the measurement noise
        self.noise_std = np.array(cfg.CAM_NOISE_STD)
        
    def can_see(self, robot_pos_arr, obstacles_map):
        """
        Checks if the line of sight between the camera and the robot is clear.
        """
        robot_point = Point(robot_pos_arr[0], robot_pos_arr[1])
        
        # Create a LineString representing the ray of light from camera to robot
        line_of_sight = LineString([self.pos, robot_point])
        
        # If the environment is empty, the camera can see everything
        if obstacles_map.is_empty:
            return True
            
        # Check intersection with obstacles.
        # We check if the intersection has an actual length to avoid false positives 
        # when the ray merely grazes a wall's corner.
        intersection = line_of_sight.intersection(obstacles_map)
        if intersection.length > 1e-5:
            return False # The line of sight is blocked by an obstacle
            
        return True # The robot is visible!

    def get_measurement(self, true_state, obstacles_map):
        """
        Returns a noisy measurement [x, y, theta] if the robot is visible.
        Returns None if the robot is hidden behind an obstacle.
        """
        if self.can_see(true_state, obstacles_map):
            # Generate Gaussian noise
            noise = np.random.normal(0, self.noise_std)
            
            # Add noise to the true state
            measured_state = true_state + noise
            
            # Normalize the measured theta to [-pi, pi]
            measured_state[2] = (measured_state[2] + np.pi) % (2 * np.pi) - np.pi
            
            return measured_state
        else:
            return None # Blind spot


class EKF:
    """
    Extended Kalman Filter for Unicycle Robot Localization.
    """
    def __init__(self, initial_state, dt):
        # State vector: [x, y, theta]
        self.x_est = np.copy(initial_state).astype(float)
        
        # Covariance matrix (Starts very confident since we spawn the robots)
        self.P = np.eye(3) * 0.01 
        
        self.dt = dt
        
        # Process Noise Q: Uncertainty in the kinematics (slipping wheels, etc.)
        self.Q = np.diag(cfg.EKF_PROCESS_NOISE_STD) ** 2
        
        # Measurement Noise R: Uncertainty of the camera
        # Usually provided by the camera specs
        self.R = np.diag(cfg.CAM_NOISE_STD) ** 2  
        
        # Measurement Matrix H: Identity matrix since we measure exactly [x, y, theta]
        self.H = np.eye(3)
        self.I = np.eye(3)

        
    def predict(self, u):
        """
        Predicts the next state using the odometry (control input [v, w]).
        (Odometry Drift happens here)
        """
        v, w = u[0], u[1]
        theta = self.x_est[2]
        
        # 1. State Prediction
        self.x_est[0] += v * np.cos(theta) * self.dt
        self.x_est[1] += v * np.sin(theta) * self.dt
        self.x_est[2] += w * self.dt
        
        # Normalize theta
        self.x_est[2] = (self.x_est[2] + np.pi) % (2 * np.pi) - np.pi
        
        # 2. Jacobian A Calculation 
        # wrt state
        A = np.array([
            [1.0, 0.0, -v * np.sin(theta) * self.dt],
            [0.0, 1.0,  v * np.cos(theta) * self.dt],
            [0.0, 0.0,                          1.0]
        ])
        
        # wrt noise 
        G = np.array([
            [np.cos(theta) * self.dt, 0.0],
            [np.sin(theta) * self.dt, 0.0],
            [0.0,                 self.dt]
        ])
        
        
        # 3. Covariance Prediction
        self.P = A @ self.P @ A.T + G @ self.Q @ G.T
        
        
    def update(self, z):
        """
        Corrects the state estimation using the camera measurement z.
        """
        if z is None:
            # If the camera can't see the robot, we skip the update phase!
            # The robot will only rely on the prediction (drift increases).
            return
            
        # 1. Innovation / Residual (Difference between measurement and prediction)
        y = z - (self.H @ self.x_est)
        
        # CRITICAL: Normalize the angular error in the residual to [-pi, pi]
        # Otherwise, 359 deg - 1 deg = 358 error, instead of -2 error!
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        
        # 2. Innovation Covariance S
        S = self.H @ self.P @ self.H.T + self.R
        
        # 3. Kalman Gain K
        W = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 4. State Update
        self.x_est = self.x_est + (W @ y)
        self.x_est[2] = (self.x_est[2] + np.pi) % (2 * np.pi) - np.pi
        
        # 5. Covariance Update
        self.P = (self.I - W @ self.H) @ self.P