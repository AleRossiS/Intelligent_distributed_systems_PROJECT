import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
import Intelligent_distributed_systems_PROJECT.config as cfg

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
    return [Polygon(obs) for obs in self.obstacles]
  
  
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
    
  