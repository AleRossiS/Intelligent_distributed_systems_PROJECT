import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import config as cfg
import numpy as np
import math

class Environment:
  """
  Class representing a 2D environment for robot simulation, including obstacles boundaries, robots, pikup zone and delivery zone.
  """
  
  def __init__(self):
    self.xlim = cfg.X_LIM
    self.ylim = cfg.Y_LIM
    self.robots = []    # List of Robot instances in the environment
    self.obstacles = []  # List of obstacles vertices (each obstacle is defined as a list of (x, y) tuples)
    self.pickup_zone = None
    self.delivery_zone = None
    self.target_object = None
    
    
  def add_robot(self, robot):
    """Adds a robot to the environment.
    args:
        robot: an instance of the Robot class to be added to the environment.
    """
    self.robots.append(robot)
    
    
  def add_obstacle(self, vertices):
    """
    Adds a generic polygonal obstacle and retu.
    Args:
        vertices: list of (x, y) tuples representing the polygon vertices.
                  e.g., [(0,0), (2,0), (2,2), (0,2)]
    """
    self.obstacles.append(vertices)
    

  def get_obstacle_polygons(self):
    """Returns a list of shapely Polygons representing the obstacles in the environment."""
    polygons = [Polygon(obs) for obs in self.obstacles]
    
    # Add the target object as an obstacle if it exists
    if self.target_object is not None:
      polygons.append(self.target_object)
    
    return polygons


  def add_walls(self, wall_thickness=0.5):
    """Adds walls around the environment boundaries as obstacles.
    Args:
        wall_thickness: thickness of the walls to be added around the environment boundaries.
    """
    x_min, y_min = self.xlim[0], self.ylim[0]
    x_max, y_max = self.xlim[1], self.ylim[1]
    
    # 1. North Wall (Top)
    self.add_obstacle([
        (x_min + wall_thickness, y_max - wall_thickness), (x_max - wall_thickness, y_max - wall_thickness), 
        (x_max - wall_thickness, y_max), (x_min + wall_thickness, y_max)
    ])
    
    # 2. South Wall (Bottom)
    self.add_obstacle([
        (x_min + wall_thickness, y_min), (x_max - wall_thickness, y_min), 
        (x_max - wall_thickness, y_min + wall_thickness), (x_min + wall_thickness, y_min + wall_thickness)
    ])
    
    # 3. East Wall (Right)
    self.add_obstacle([
        (x_max - wall_thickness, y_min), (x_max, y_min), 
        (x_max, y_max), (x_max - wall_thickness, y_max)
    ])
    
    # 4. West Wall (Left)
    self.add_obstacle([
        (x_min, y_min), (x_min + wall_thickness, y_min), 
        (x_min + wall_thickness, y_max), (x_min, y_max)
    ])
    
    
  def add_target_object(self, cx, cy, outer_radius=0.4, inner_radius=0.15):
      """
      Adds a target object (star shape) as a Shapely Polygon to be detectable by LiDAR.
      
      Args:
          cx: x coordinate of the star center.
          cy: y coordinate of the star center.
          outer_radius: distance from center to the outer points of the star.
          inner_radius: distance from center to the inner corners of the star.
      """
      num_points = 5
      vertices = []
      
      # Angle step to alternate between outer and inner points of the star
      angle_step = np.pi / num_points 

      # Generate vertices for the star shape 
      for i in range(2 * num_points):
          
          r = outer_radius if i % 2 == 0 else inner_radius
          
          angle = np.pi / 2 - i * angle_step
          
          x = cx + r * np.cos(angle)
          y = cy + r * np.sin(angle)
          vertices.append((x, y))

      # Build the star-shaped target object as a Shapely Polygon
      self.target_object = Polygon(vertices)
      
      
  def get_treasure_signal_polygon(self):
    """
    Computes the signal area polygon for the treasure, accounting for obstacle occlusions.
    """
    if self.target_object is None:
      return Polygon()

    # Treasure position and signal range
    tx_pos = (self.target_object.centroid.x, self.target_object.centroid.y)
    tx_point = Point(tx_pos)
    sig_range = cfg.BL_RANGE 
    
    # Merge obstacles into a single geometry map (excluding the treasure itself)
    obstacles = [Polygon(obs) for obs in self.obstacles]
    if obstacles:
      collision_map = unary_union(obstacles).buffer(0)
    else:
      collision_map = Polygon()
        
    # Base rays generation
    angles_list = list(np.linspace(0, 2 * np.pi, 100))
    
    # Target obstacle vertices to avoid corner-cutting 
    for obs in obstacles:
      for x, y in obs.exterior.coords:
        angle = math.atan2(y - tx_pos[1], x - tx_pos[0])
        # Normalize angle to [0, 2*pi]
        angle = angle % (2 * np.pi)
        angles_list.extend([angle - 0.0001, angle, angle + 0.0001])
            
    # Sort normalized angles counter-clockwise to prevent polygon self-intersection
    angles = sorted([a % (2 * np.pi) for a in angles_list])
    
    ray_vertices = []
    
    # Cast rays and compute intersections with obstacles
    for angle in angles:
      dx = np.cos(angle) * sig_range
      dy = np.sin(angle) * sig_range
      ray_end = (tx_pos[0] + dx, tx_pos[1] + dy)
      
      ray_line = LineString([tx_pos, ray_end]) 
      
      # Subtract obstacles from the ray
      if not collision_map.is_empty:
        visible_ray = ray_line.difference(collision_map)
      else:
        visible_ray = ray_line 
          
      # Extract the valid end point of the ray
      if not visible_ray.is_empty:
        if visible_ray.geom_type == 'MultiLineString':
          # Select the segment starting closest to the treasure
          best_line = min(visible_ray.geoms, key=lambda line: Point(line.coords[0]).distance(tx_point))
          
          # If the segment starts far away, the ray is blocked from the source
          if Point(best_line.coords[0]).distance(tx_point) > 0.05:
            hit_point = tx_pos
          else:
            hit_point = best_line.coords[-1]
              
        elif visible_ray.geom_type == 'LineString':
          if Point(visible_ray.coords[0]).distance(tx_point) > 0.05:
            hit_point = tx_pos
          else:
            hit_point = visible_ray.coords[-1]
        else:
          hit_point = tx_pos
      else:
        # Ray is completely blocked
        hit_point = tx_pos
        
      ray_vertices.append(hit_point)
    
    # Fallback for degenerate polygons
    if len(ray_vertices) < 3:
      return tx_point.buffer(0.01)
        
    return Polygon(ray_vertices)

  
  def render(self, ax):
    """Renders the environment on a given Matplotlib axis.
    args:
        ax: a Matplotlib axis object where the environment will be drawn.
    """
    ax.clear()
    ax.set_xlim(self.xlim)
    ax.set_ylim(self.ylim)
    ax.set_aspect('equal')

    # Draw Obstacles
    for obs_vertices in self.obstacles:
      poly_patch = patches.Polygon(obs_vertices, closed=True, color='tab:gray', alpha=1)
      ax.add_patch(poly_patch)

    # Draw Robots
    for r in self.robots:
      r.draw(ax)
      
    # Draw Target Object
    if self.target_object is not None:
        # Vertices coordinates of the target object polygon
        target_coords = list(self.target_object.exterior.coords)
        
        # Build a star-shaped patch for visualization
        star_patch = patches.Polygon(
            target_coords, 
            closed=True, 
            facecolor='gold', 
            edgecolor='darkorange', 
            linewidth=1.5, 
            zorder=5
        )
        ax.add_patch(star_patch)
        
    # Draw Treasure Signal Area
    sig_area = self.get_treasure_signal_polygon()
      
    if not sig_area.is_empty and sig_area.geom_type == 'Polygon':
        sig_coords = list(sig_area.exterior.coords)
        sig_patch = patches.Polygon(
            sig_coords, 
            closed=True, 
            facecolor='cyan', 
            edgecolor='deepskyblue', 
            linestyle='--', 
            linewidth=1,
            alpha=0.15,
            zorder=2
        )
        ax.add_patch(sig_patch)

        x, y = zip(*sig_coords) 
        ax.plot(x, y, color='deepskyblue', linestyle='-', linewidth=0.5, zorder=3)
