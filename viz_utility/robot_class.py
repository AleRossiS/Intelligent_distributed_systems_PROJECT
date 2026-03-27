import numpy as np
import matplotlib.pyplot as plt
import random
import config as cfg
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union


class Robot:
  """
  A simple unicycle robot class for 2D simulation
  """ 
  
  def __init__(self, robot_id, x = 0.0, y = 0.0, theta = 0.0, radius = cfg.ROBOT_RADIUS, color = 'blue'):
    
    # Robot attributes
    self.robot_id = robot_id
    self.state = np.array([x, y, theta], dtype=float) # [x, y, theta]
    self.radius = radius
    self.color = color
    
    # Robot trajectory history for visualization
    self.history_x = [self.state[0]]
    self.history_y = [self.state[1]]
    
    self.other_robots_positions = {} # Dictionary to store positions of other robots {robot_id: (x, y)}
    self.id_to_call = None # ID of the robot to call for help when treasure is found
    self.rendezvous_pos = None # Position where the robot will meet others
    self.is_finder = False # Flag to indicate if this robot is the one that found the treasure
    self.relocation_done = False # Flag to indicate if the robot has already relocated to P1 after finding the treasure

    # State mode paraters
    self.state_mode = "SEARCHING" # can be "SEARCHING", "INSPECTING", "TRASURE_FOUND"
    self.found_treasure = False
    self.target_treasure_pos = None

    # Memory of inspected points and shared map 
    self.inspected_points = [] # List of (x, y) tuples representing points that this robot has inspected
    self.shared_map = [] # List of (x, y) tuples representing the combined map of all robots 
    
    # Data broadcasting parameters
    offset = cfg.TX_INTERVAL_TICKS // cfg.NUM_ROBOTS
    self.tx_clock = ((self.robot_id - 1) * offset) % cfg.TX_INTERVAL_TICKS # clk time at which the robot will transmit 
    self.is_transmitting = False # Flag to indicate if the robot is currently transmitting data
    
    
  def update_pose(self, u, dt):
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
    
    # State derivative
    self.state_dot = np.array([
        u[0] * np.cos(self.state[2]),  # x_dot
        u[0] * np.sin(self.state[2]),  # y_dot
        u[1]                           # theta_dot
    ]) 
    
    # State update: Euler integration
    self.state += self.state_dot * dt
    
    #NB: i've just did:
    # x_k+1 = x_k + x_dot * dt
    # y_k+1 = y_k + y_dot * dt
    # theta_k+1 = theta_k + theta_dot * dt
    
    # Normalize theta to be within [-pi, pi]
    self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
    
    # Save resulting position to the history lists
    self.history_x.append(self.state[0])
    self.history_y.append(self.state[1])
    
    
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
        'sender_pos': (self.state[0], self.state[1]), # Share current X, Y position # TODO: METTERR EKF ESTIM
        'map_points': self.inspected_points,          # Share explored map
        
        # --- FUTURE EXPANSION (Treasure Consensus) ---
        # 'consensus_H': self.H if hasattr(self, 'H') else None,
        # 'consensus_b': self.b if hasattr(self, 'b') else None,
    }
      return payload


  def receive_payload(self, payload):
    """
    Processes incoming data from another robot and updates internal state accordingly.
    
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
    
    
  def measure_target_edge(self, target_obj, obstacles_map):
    """
    Simulates the directional sensor and LiDAR to detect the target position.
    Uses a ray-casting approach based on the direction provided by the directional sensor.
    
    Args:
      target_obj: Shapely Polygon representing the target.
      obstacles_map: A Shapely Polygon representing the collision map.
      
    Returns:
      True if the target is detected and measured, False otherwise.
    """
    
    # 1. Check if target exists
    if target_obj is None:
      return False

    robot_pos = Point(self.state[0], self.state[1])
    target_center = target_obj.centroid
    
    
    # 2. Directional Sensor: Get the angle towards the target
    dx = target_center.x - robot_pos.x
    dy = target_center.y - robot_pos.y
    dist_to_center = np.hypot(dx, dy)
    
    # Check if we are within the maximum broadcast/signal range
    if dist_to_center > cfg.BL_RANGE * 0.9: 
      return False

    # The directional sensor gives us the angle
    target_angle = np.arctan2(dy, dx)
    
    
    # 3. Simulate LiDAR: Cast a ray in the sensed direction up to the sensing range and check for obstacles 
    ray_end = (robot_pos.x + np.cos(target_angle) * cfg.SENSING_RANGE, 
                robot_pos.y + np.sin(target_angle) * cfg.SENSING_RANGE)
    
    # Line representing the LiDAR ray of length equal to the sensing range in the direction of the target
    ray_line = LineString([robot_pos, ray_end])
    
    
    # 4. Build Collision Map (MODIFICATO: non serve più fondere nulla!)
    collision_map = obstacles_map 
        
        
    # 5. Calculate visible ray by subtracting the collision map from the ray line
    if not collision_map.is_empty:
      visible_ray = ray_line.difference(collision_map) # = Part of the ray that is not occluded by obstacles
    else:
      visible_ray = ray_line
        
    # 6. Find the hit point of the visible ray
    hit_point = None
    
    # This part handles the case where the visible ray is split into multiple segments due to obstacles
    if not visible_ray.is_empty:
      if visible_ray.geom_type == 'MultiLineString':
        for line in visible_ray.geoms:
          if line.distance(robot_pos) < 1e-5:
            hit_point = Point(line.coords[-1])
            break
      elif visible_ray.geom_type == 'LineString':
        hit_point = Point(visible_ray.coords[-1])
            
    if hit_point is None:
      return False 
        
    # 7. Check if the hit point is the target edge (using a small threshold to account for numerical issues)
    if hit_point.distance(target_obj) < 1e-3: 
        
      # 8. Update internal state
      self.state_mode = "TREASURE_FOUND"
      self.found_treasure = True
      self.target_treasure_pos = np.array([hit_point.x, hit_point.y])
      self.is_finder = True
      self.has_called_help = False
      
      return True
        
    return False # If we hit something else or nothing at all, the target is not detected

  def broadcast_help(self, all_robots):
    """
    Broadcasts the target position to all other robots if this robot is the finder.
    Ensures the broadcast happens only once to avoid spamming the communication channel.
    
    Args:
      all_robots: List containing all Robot instances in the environment.
    """
        
    # 1. Call all other robots
    for peer in all_robots:
      if peer.robot_id != self.robot_id: # Do not call itself
          
        # Only interrupt robots that are not already dealing with the treasure
        if peer.state_mode not in "TREASURE_FOUND":
          peer.state_mode = "TREASURE_FOUND"
          peer.target_treasure_pos = self.target_treasure_pos
          print(f"[RADIO] Robot {self.robot_id} called Robot {peer.robot_id} to join the formation!")
    
      # 2. Set the flag to indicate help has been called
      self.has_called_help = True

    
  def draw(self, ax, show_history=True):
    """
    Draws the robot and its trajectory on a given Matplotlib axis
    
    Args:
      ax: Matplotlib axis to draw on
      show_history: if True, also draws the trajectory history of the robot
    """
    x, y, theta = self.state
    
    # draw the robot as a circle with a line indicating the front direction
    circle = plt.Circle((x, y), self.radius, 
                        facecolor=self.color, 
                        edgecolor='black', 
                        linewidth=1, 
                        fill=True, 
                        alpha=1)
    ax.add_patch(circle)
    
    front_x = x + self.radius * np.cos(theta)
    front_y = y + self.radius * np.sin(theta)
    ax.plot([x, front_x], [y, front_y], color='black', linewidth=2)
    
    # Write robot ID and state mode above the robot
    label = f"R{self.robot_id}\n{self.state_mode}"
    ax.text(x, y + self.radius * 1.4, label, ha='center', va='bottom', fontsize=8, color='black')
    
    # Plot robot trajectory
    if show_history:
      ax.plot(self.history_x, self.history_y, color=self.color, linewidth=1.5, alpha=0.4, linestyle='-')
    
    
    ### THIS ALLOWS TO SEE THE INSPECTED POINTS OF EACH ROBOT (USE WITH CAUTION, IT CAN GET MESSY) ###
    # Draw inspected points
    # if len(self.inspected_points) > 0:
    #     pts = np.array(self.inspected_points)
    #     ax.scatter(pts[:, 0], pts[:, 1], color=self.color, s=15, marker='x', alpha=0.8)
        
        
    # # Draw shared map points (if any) with a smaller marker and some transparency
    # if len(self.inspected_points) > 0:
    #     pts = np.array(self.inspected_points)
    #     ax.scatter(pts[:, 0], pts[:, 1], color=self.color, s=8, alpha=0.6, zorder=1)
    
    
    ### TEST  TO SEE ONLY WHAT ROBOT 1 SEES ###
    if self.robot_id == 1 and hasattr(self, 'shared_map') and len(self.shared_map) > 0:
      pts = np.array(self.shared_map)
      ax.scatter(pts[:, 0], pts[:, 1], color='black', s=20, marker='x', alpha=1.0, zorder=3)

    # Draw transmission indicator if the robot is currently transmitting
    if getattr(self, 'is_transmitting', False):
      # Draw expanding circles to represent the transmission signal
      radii = [self.radius * 2.0, self.radius * 3.5]
      alphas = [0.6, 0.3] # Corresponding alpha values for the circles
      
      for r_wave, a_wave in zip(radii, alphas):
        tx_circle = plt.Circle((x, y), r_wave, color='magenta', fill=False, linestyle='-', linewidth=0.5, alpha=a_wave)
        ax.add_patch(tx_circle)
        
      # text indicator for transmission TX
      ax.text(x, y + self.radius * 3.75, "TX", ha='center', va='bottom', fontsize=10, color='magenta', fontweight='bold')
      
    # Draw the target position if known 
    if self.target_treasure_pos is not None:
      ax.scatter(self.target_treasure_pos[0], self.target_treasure_pos[1], 
                 color=self.color, s=80, marker='o', edgecolors='black', zorder=10, alpha=0.5)

  