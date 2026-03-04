import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon

class Environment:
  """
  Class representing a 2D environment for robot simulation, including obstacles boundaries, robots, pikup zone and delivery zone.
  """
  
  def __init__(self, xlim=(-10, 10), ylim=(-10, 10)):
    self.xlim = xlim
    self.ylim = ylim
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

    
  def set_zones(self, pickup_rect, delivery_rect):
    """Sets the pickup and delivery zones.
    args:
        pickup_rect: tuple (x, y, width, height) defining the pickup zone.
        delivery_rect: tuple (x, y, width, height) defining the delivery zone.
    """
    self.pickup_zone = pickup_rect
    self.delivery_zone = delivery_rect
  
  
  def render(self, ax):
    """Renders the environment on a given Matplotlib axis.
    args:
        ax: a Matplotlib axis object where the environment will be drawn.
    """
    ax.clear()
    ax.set_xlim(self.xlim)
    ax.set_ylim(self.ylim)
    ax.set_aspect('equal')
    
    # Draw Pickup and Delivery Zones
    if self.pickup_zone:
      x, y, w, h = self.pickup_zone
      ax.add_patch(patches.Rectangle((x, y), w, h, color='green', alpha=0.2))
      ax.text(x+w/2, y+h/2, "PICKUP", color='green', ha='center', alpha=0.5)

    if self.delivery_zone:
      x, y, w, h = self.delivery_zone
      ax.add_patch(patches.Rectangle((x, y), w, h, color='orange', alpha=0.2))
      ax.text(x+w/2, y+h/2, "DELIVERY", color='orange', ha='center', alpha=0.5)

    # Draw Obstacles
    for obs_vertices in self.obstacles:
      poly_patch = patches.Polygon(obs_vertices, closed=True, color='gray', alpha=0.8)
      ax.add_patch(poly_patch)

    # Draw Robots
    for r in self.robots:
      r.draw(ax)
    
  