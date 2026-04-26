import numpy as np
import matplotlib.pyplot as plt
import random
import config as cfg
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

from ekf_localization import EKF

class Robot:
  """
  A simple unicycle robot class for 2D simulation
  """ 
  
  def __init__(self, robot_id, x = 0.0, y = 0.0, theta = 0.0, radius = cfg.ROBOT_RADIUS, color = 'blue', dt=0.1):
    
    self.true_state = np.array([x, y, theta], dtype=float) # True state for simulation purposes (not accessible to the robot)
    
    # EKF initial state = camera measurement
    cam_noise = np.random.normal(0, cfg.CAM_NOISE_STD)
    self.state = self.true_state + cam_noise #EKF state initial estimate (= initial camera measurement with noise) 
    self.robot_id = robot_id
    self.ekf = EKF(initial_state=self.state, dt=dt) # EKF instance for localization
    self.radius = radius
    self.color = color
    
    # Robot trajectory history for visualization
    self.history_true_x = [self.true_state[0]]
    self.history_true_y = [self.true_state[1]]
    self.history_est_x = [self.state[0]]
    self.history_est_y = [self.state[1]]
    
    self.other_robots_positions = {} # Dictionary to store positions of other robots {robot_id: (x, y)}
    self.id_to_call = None # ID of the robot to call for help when treasure is found
    self.rendezvous_pos = None # Position where the robot will meet others
    self.is_finder = False # Flag to indicate if this robot is the one that found the treasure
    self.relocation_done = False # Flag to indicate if the robot has already relocated to P1 after finding the treasure

    # State mode parameters
    self.state_mode = "SEARCHING" # can be "SEARCHING", "INSPECTING", "TREASURE_FOUND"
    self.found_treasure = False
    self.target_treasure_pos = None
    self.mapping_dir = None # Direction for mapping movement (1 or -1) that is clockwise or counterclockwise

    # Memory of inspected points and shared map 
    self.inspected_points = [] # List of (x, y) tuples representing points that this robot has inspected
    self.shared_map = [] # List of (x, y) tuples representing the combined map of all robots 
    
    # Data broadcasting parameters
    offset = cfg.TX_INTERVAL_TICKS // cfg.NUM_ROBOTS
    self.tx_clock = ((self.robot_id - 1) * offset) % cfg.TX_INTERVAL_TICKS # clk time at which the robot will transmit 
    self.is_transmitting = False # Flag to indicate if the robot is currently transmitting data

    self.consensus_treasure_pos = None
    self.target_treasure_pos = None
    self.gamma = 0.5
    self.has_sensed_treasure = False
    self.last_measure_time = 0.0
    self.measure_interval = 1.0
    self.local_measure_count = 1
    
    
  def update_pose(self, u, dt, camera_measurement=None):
    """
    Updates the pose using the unicycle model:
    
      x_ = v * cos(theta)
      y_dot = v * sin(theta)
      theta_dot = w
      
    And saves the new position in the history lists for later visualization.

    Args:
      u: control input [v, w]
      dt: time step
    """

    # Sinulate odometry noise (this is where the odometry drift happens in the simulation)
    noise_v = np.random.normal(0, cfg.SIGMA_V)  
    noise_w = np.random.normal(0, cfg.SIGMA_W) 

    v_real = u[0] + noise_v
    w_real = u[1] + noise_w

    self.true_state[0] += v_real * np.cos(self.true_state[2]) * dt
    self.true_state[1] += v_real * np.sin(self.true_state[2]) * dt
    self.true_state[2] += w_real * dt
    self.true_state[2] = (self.true_state[2] + np.pi) % (2 * np.pi) - np.pi

    # EKF Prediction e update Step
    self.ekf.predict(u) # EKF prediction step using the control input (odometry)
    self.ekf.update(camera_measurement) # EKF update step using the camera measurement (if available)

    self.state = np.copy(self.ekf.x_est) # Update the robot's estimated state with the EKF estimate

    # Save true and estimated positions to the history lists
    self.history_true_x.append(self.true_state[0])
    self.history_true_y.append(self.true_state[1])
    self.history_est_x.append(self.state[0])
    self.history_est_y.append(self.state[1])
    
    
    # State derivative
    self.state_dot = np.array([
        u[0] * np.cos(self.state[2]),  # x_dot
        u[0] * np.sin(self.state[2]),  # y_dot
        u[1]                           # theta_dot
    ]) 
    
    # # State update: Euler integration
    # self.state += self.state_dot * dt
    
    # #NB: i've just did:
    # # x_k+1 = x_k + x_dot * dt
    # # y_k+1 = y_k + y_dot * dt
    # # theta_k+1 = theta_k + theta_dot * dt
    
    # # Normalize theta to be within [-pi, pi]
    # self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
    
    # # Save resulting position to the history lists
    # self.history_x.append(self.state[0])
    # self.history_y.append(self.state[1])
    
    
  def tick_timer(self):
    """
    Decrements the internal transmission timer.
    Returns True if the timer expires (meaning the robot wants to transmit), 
    and resets it.
    
    returns:
      bool: True if it's time to transmit, False otherwise
    """
    self.tx_clock += 1
    
    if self.tx_clock >= cfg.TX_INTERVAL_TICKS:
        self.tx_clock = 0  
        return True
        
    return False
      
      
  def update_shared_map(self, incoming_map, threshold=cfg.MAP_RESOLUTION):
    """
    Updates the global shared map with new points from incoming data.
    Also performs distance checking to avoid duplicate points or points that are too close.
    
    Args:
      incoming_map: list of (x, y) tuples representing points from another robot's map
      threshold: minimum Euclidean distance to consider a new point as unique
    """
    
    # If the received packet is empty, do nothing
    if not incoming_map:
      return
        
    # If the robot's shared map is currently empty, copy all incoming points directly
    if not self.shared_map:
      # We convert points to tuples to keep the data structure consistent
      self.shared_map.extend([tuple(p) for p in incoming_map])
      return

    # Convert the current shared map to a NumPy array for fast vectorized operations
    shared_map_array = np.array(self.shared_map)
    
    for new_point in incoming_map:
      # Calculate the Euclidean distance between the new point and ALL existing points in the shared map
      distances = np.linalg.norm(shared_map_array - np.array(new_point), axis=1)
      
      # If the closest point is further than the threshold, it is a valid new point
      if np.min(distances) >= threshold:
        # Append as a tuple
        self.shared_map.append(tuple(new_point))
        
        # Update the NumPy array for the next iterations
        shared_map_array = np.vstack([shared_map_array, new_point])
        
              
  def generate_payload(self):
      """
      Creates a dictionary containing all the data the robot wants to share 
      with the swarm over the network.
      
      Returns:
        dict: The data payload.
      """
      payload = {
        'sender_id': self.robot_id,
        'sender_pos': (self.state[0], self.state[1]), # Share current X, Y position estimated by the EKF
        'map_points': self.inspected_points,          # Share explored map
        'treasure_est': getattr(self, 'consensus_treasure_pos', None) # Share treasure position estimate 
    }
      return payload


  def receive_payload(self, payload):
    """
    Processes incoming data from another robot and updates internal state accordingly.
    Also performs consensus update for the treasure position if the incoming payload contains an estimate.
    
    Args:
      payload: dict containing the data sent by another robot
    """
    
    sender_id = payload.get('sender_id')
    
    # 1. Update positions
    sender_pos = payload.get('sender_pos')
    if sender_id is not None and sender_pos is not None:
      self.other_robots_positions[sender_id] = sender_pos

    # 2. Update the map with the received point using the merging function defined above
    incoming_map = payload.get('map_points')
    self.update_shared_map(incoming_map)

    incoming_treasure = payload.get('treasure_est')
    if incoming_treasure is not None:
      if getattr(self, 'consensus_treasure_pos', None) is None:
        self.consensus_treasure_pos = np.copy(incoming_treasure)
        self.target_treasure_pos = self.consensus_treasure_pos
        self.state_mode = "TREASURE_FOUND"
      else:
        self.consensus_treasure_pos = (1.0 - self.gamma) * self.consensus_treasure_pos + (self.gamma * incoming_treasure)
        self.target_treasure_pos = self.consensus_treasure_pos
    
    
  
    
  def draw(self, screen, world_to_pixel_func, font=None, camera_status=None, camera_pos=None):
    """
    Draws the robot, its trajectory, and status text using Pygame.
    
    Args:
      screen: Pygame surface to draw on.
      world_to_pixel_func: Function to convert world coordinates to pixel coordinates.
      font: Pygame font object for rendering text (optional).
      camera_status: Optional string indicating the camera status to display.
    """
    import pygame 
    
    # Map matplotlib string colors to RGB
    color_map = {
        'tab:blue': (31, 119, 180),
        'tab:red': (214, 39, 40),
        'tab:green': (44, 160, 44),
        'tab:orange': (255, 127, 14),  
        'tab:purple': (148, 103, 189),  
        'black': (0, 0, 0)
    }
    color_rgb = color_map.get(self.color, (31, 119, 180)) # Default blue
    
    # Get pixel coordinates
    x, y, theta = self.true_state # Use true_state for physical drawing
    tx, ty = world_to_pixel_func(x, y)
    
    # Calculate dynamic radius in pixels
    test_x, _ = world_to_pixel_func(x + self.radius, y)
    radius_px = abs(test_x - tx)

    # 1. Draw trajectory history (limit to last 200 points to save FPS)
    if len(self.history_true_x) > 1:
        pts = [world_to_pixel_func(hx, hy) for hx, hy in zip(self.history_true_x[-200:], self.history_true_y[-200:])]
        if len(pts) > 1:
            pygame.draw.lines(screen, color_rgb, False, pts, 2)
            
    # 2. Draw Estimated Position (EKF)
    # Display where the robot "thinks" it is based on the EKF state (self.state)
    ex, ey = self.state[0], self.state[1]
    px_est, py_est = world_to_pixel_func(ex, ey)
    
    # Use a light gray, empty circle to represent the estimated position
    pygame.draw.circle(screen, (150, 150, 150), (px_est, py_est), radius_px, 2)

    # 3. Draw robot body
    pygame.draw.circle(screen, color_rgb, (tx, ty), radius_px)
    pygame.draw.circle(screen, (0, 0, 0), (tx, ty), radius_px, 3) # Black outline

    # 4. Draw orientation line
    end_x = x + self.radius * np.cos(theta)
    end_y = y + self.radius * np.sin(theta)
    px_end, py_end = world_to_pixel_func(end_x, end_y)
    pygame.draw.line(screen, (0, 0, 0), (tx, ty), (px_end, py_end), 3)

    # 5. Draw transmission waves (single fast expansion animation)
    
    # Initialize animation variables if they don't exist
    if not hasattr(self, 'wave_radius'):
        self.wave_radius = radius_px
        self.wave_active = False
        self.prev_tx = False # Tracks previous frame transmission state

    current_tx = getattr(self, 'is_transmitting', False)

    # Trigger or restart the wave animation on a NEW transmission pulse
    if current_tx and not self.prev_tx:
        self.wave_active = True
        self.wave_radius = radius_px # Reset radius to start a new wave

    self.prev_tx = current_tx # Save state for the next frame

    # Draw and expand the wave if animation is active
    if self.wave_active:
        max_wave_radius = radius_px * 2.5  # Very small maximum wave size for a quick burst
        expansion_speed = 6                # High expansion speed for a fast animation
        
        # Calculate fading effect (alpha drops from 255 to 0)
        fade_ratio = (self.wave_radius - radius_px) / (max_wave_radius - radius_px)
        fade_ratio = max(0.0, min(1.0, fade_ratio)) # Clamp between 0 and 1
        alpha = int(255 * (1 - fade_ratio))
        
        # Temporary surface for alpha channel
        temp_surf = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        
        # Single wave pulse
        pygame.draw.circle(temp_surf, (255, 0, 255, alpha), (tx, ty), int(self.wave_radius), 2)
        
        screen.blit(temp_surf, (0, 0))
        
        # Expand the wave for the next frame
        self.wave_radius += expansion_speed
        
        # Stop animation and reset when the wave reaches max radius
        if self.wave_radius > max_wave_radius:
            self.wave_active = False
            self.wave_radius = radius_px

    # 6. Draw text info (State and Camera status)
    if font is not None:
        # Robot state mode
        state_text = f"R{self.robot_id} {self.state_mode}"
        text_surf = font.render(state_text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=(tx, ty - radius_px - 25))
        pygame.draw.rect(screen, (255, 255, 255), text_rect.inflate(4, 2)) # White background
        screen.blit(text_surf, text_rect)
        
        # Camera status (if provided)
        if camera_status:
            cam_color = (0, 200, 0) if "TRACKING" in camera_status else (200, 0, 0)
            cam_surf = font.render(camera_status, True, cam_color)
            cam_rect = cam_surf.get_rect(center=(tx, ty - radius_px - 10))
            pygame.draw.rect(screen, (255, 255, 255), cam_rect.inflate(4, 2))
            screen.blit(cam_surf, cam_rect)
              
    # 7. Draw mapped points
    
    # --- VISUALIZATION FLAG ---
    # True = Show the 'inspected_points' of ALL robots (Individual mapping)
    # False = Show only the 'shared_map' of Robot 1 (Consensus mapping)
    SHOW_ALL_INSPECTED_POINTS = True 
    
    size = 3
    thickness = 2

    if SHOW_ALL_INSPECTED_POINTS:
        # Display points inspected individually by each robot
        if hasattr(self, 'inspected_points') and len(self.inspected_points) > 0:
            for pt in self.inspected_points:
                px, py = world_to_pixel_func(pt[0], pt[1])

                # Use a dark gray color
                pygame.draw.line(screen, (0, 0, 0), (px - size, py - size), (px + size, py + size), thickness)
                pygame.draw.line(screen, (0, 0, 0), (px - size, py + size), (px + size, py - size), thickness)
                
    else:
        # Display only the shared, fused map of Robot 1 (Useful for consensus debugging)
        if self.robot_id == 1 and hasattr(self, 'shared_map') and len(self.shared_map) > 0:
            for pt in self.shared_map:
                px, py = world_to_pixel_func(pt[0], pt[1])

                # Use black color
                pygame.draw.line(screen, (0, 0, 0), (px - size, py - size), (px + size, py + size), thickness)
                pygame.draw.line(screen, (0, 0, 0), (px - size, py + size), (px + size, py - size), thickness)
                
  
    # 8. Draw Line of Sight to the Global Camera
    # If the camera is tracking the robot, draw a line connecting them
    if camera_status == "TRACKING" and camera_pos is not None:
        c_x, c_y = camera_pos[0], camera_pos[1]
        cam_px = world_to_pixel_func(c_x, c_y)
        
        # Thin green line to indicate active tracking
        pygame.draw.line(screen, (255, 0, 255), (tx, ty), cam_px, 1)