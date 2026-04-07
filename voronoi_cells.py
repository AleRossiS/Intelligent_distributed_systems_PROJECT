import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import polygons
from shapely import geometry
import shapely
from shapely.geometry import Polygon, box, point, MultiPoint, LineString, Point
import config as cfg
from shapely.ops import unary_union



class VoronoiCell:
  """
  Class that handles the Voronoi cell of a robot, 
  given the positions of all robots and the environment boundaries.
  
  Attributes:
    bounds: list of the form [x_min, y_min, x_max, y_max] defining the environment boundaries.
    boundary_poly: shapely Polygon representing the environment boundaries.
  """
  
  def __init__(self):
    self.bounds = [cfg.X_LIM[0], cfg.Y_LIM[0], cfg.X_LIM[1], cfg.Y_LIM[1]]
    self.boundary_poly = box(*self.bounds) # Create a rectangle for the environment boundaries (using shapely library)

    #Ray casting resolituon 
    self.num_rays = int(360 / cfg.RAY_CASTING_RESOLUTION)
    self.angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False) # Precompute the angles for ray casting

  def ray_casting_polygon(self, true_state, est_state, collision_map):
      """
      Compute the visibility polygon for a robot simulating ray casting.
      
      Args:
          true_state: array/list [x, y, theta] representing the REAL state of the robot.
          est_state: array/list [x, y, theta] representing the ESTIMATED state (EKF).
          collision_map: shapely Polygon representing the union of all obstacles.
      
      Returns:
          A shapely Polygon representing the visibility area mapped by the robot's belief.
      """
      # Estraiamo le coordinate
      true_x, true_y, true_theta = true_state
      est_x, est_y, est_theta = est_state
      
      ray_vertices = [] 
      robot_point_true = Point(true_x, true_y) 
      
      # 1. Cast rays in all directions
      # self.angles ora rappresenta l'angolo del raggio "relativo" al robot
      for local_angle in self.angles:
          
          # Sensing point in the true world 
          true_ray_angle = true_theta + local_angle
          
          dx_true = np.cos(true_ray_angle) * cfg.SENSING_RANGE
          dy_true = np.sin(true_ray_angle) * cfg.SENSING_RANGE
          ray_end_true = (true_x + dx_true, true_y + dy_true)
          
          # Create a line representing the ray from the robot to the end point
          ray_line_true = LineString([(true_x, true_y), ray_end_true]) 
          
          # Cut from the ray line the parts that intersect with the collision map 
          if not collision_map.is_empty:
              visible_ray = ray_line_true.difference(collision_map)
          else:
              visible_ray = ray_line_true 
          
          hit_point_true = ray_end_true
          
          if not visible_ray.is_empty:
              if visible_ray.geom_type == 'MultiLineString':
                  for line in visible_ray.geoms:
                      if line.distance(robot_point_true) < 1e-5:
                          hit_point_true = line.coords[-1] 
                          break
              elif visible_ray.geom_type == 'LineString':
                  hit_point_true = visible_ray.coords[-1]
          
          # Calculate the distance from the robot to the hit point in the true world (this simulates the actual sensor measurement)
          measured_dist = np.sqrt((hit_point_true[0] - true_x)**2 + (hit_point_true[1] - true_y)**2)
          
          
          # Compute the estimated hit point in the robot's belief (using the estimated state and the measured distance)
          est_ray_angle = est_theta + local_angle
          
          est_hit_x = est_x + measured_dist * np.cos(est_ray_angle)
          est_hit_y = est_y + measured_dist * np.sin(est_ray_angle)
          
          # Add the hit point to the list of vertices for the visibility polygon
          ray_vertices.append((est_hit_x, est_hit_y))
      
      
      # 2. Build the visibility polygon from the ray vertices
      if len(ray_vertices) < 3:
          # Fallback centrato sulla posizione stimata!
          return Point(est_x, est_y).buffer(0.01)
          
      vis_poly = Polygon(ray_vertices) 
      
      return vis_poly


  def compute_and_clip(self, true_state, est_state, estim_points, robot_index, obstacles=Polygon()):
      """Compute the Voronoi cell for the given robot and clip the cells to the sensing range 
      and obstacles (including the environment boundaries) using a ray casting approach.

      Args:
          true_state: array [x, y, theta] representing the REAL state of the robot.
          est_state: array [x, y, theta] representing the ESTIMATED state of the robot.
          estim_points: list of (x, y) tuples representing all robots estimated positions by the EKF.
          robot_index: index of the robot for which we want to compute the Voronoi cell.
          obstacles: shapely Polygon representing the environment boundaries.

      Returns:
          shapely Polygon representing the clipped Voronoi cell for the specified robot.
          Centroid of the cell as a numpy array [cx, cy].
      """
      
      ### 1. COMPUTE VORONOI CELL ### 
      # Def a very large distance to handle infinite vertices of the cells
      far_away = max(self.bounds) * 100
      
      # Create dummy robot positons as a trick to ensure all Voronoi cells are finite
      dummies = np.array([
          [self.bounds[0] - far_away, self.bounds[1] - far_away], # left bottom
          [self.bounds[2] + far_away, self.bounds[1] - far_away], # right bottom
          [self.bounds[0] - far_away, self.bounds[3] + far_away], # left top
          [self.bounds[2] + far_away, self.bounds[3] + far_away]  # right top
      ])
      
      # Combine the actual robot positions with the dummy points
      all_points = np.vstack((estim_points, dummies))
      
      # Computes the Voronoi diagram for all the points (robots + dummies)
      vor = Voronoi(all_points) # From scipy library
      
      # Extract the Voronoi region corresponding to the specified robot index
      region_idx = vor.point_region[robot_index]
      region_vertices_indices = vor.regions[region_idx]
      
      # Extract the robot position (just x, y) for the geometric checks
      robot_pos = (est_state[0], est_state[1])
      
      # Create a polygon from the vertices of the Voronoi region
      if -1 not in region_vertices_indices and len(region_vertices_indices) > 0:
        
        # 1. Voronoi polygons computation   
        vor_poly = Polygon([vor.vertices[v] for v in region_vertices_indices])
        vor_poly = vor_poly.buffer(-cfg.DIST_BTW_ROBOTS)
          
        # 2. Ray Casting (QUI AVVIENE LA MAGIA DELLA SLAM)
        vis_poly = self.ray_casting_polygon(true_state, est_state, obstacles)
        
        # 3. Clipping Voronoi cell with visibility polygon 
        current_poly = vor_poly.intersection(vis_poly)
        
        # 4. Handles the case of multiple disjoint polygons
        if current_poly.geom_type == 'MultiPolygon':
          robot_point = Point(robot_pos)
          for part in current_poly.geoms:
            if part.intersects(robot_point) or part.distance(robot_point) < 1e-5:
              current_poly = part
              break
        
        clipped_poly = current_poly
      
      # Invalid Voronoi cell case
      else:
        clipped_poly = None
        
      ### 2. COMPUTE CENTROID ###
      if clipped_poly is not None and not clipped_poly.is_empty:
        centroid = np.array([clipped_poly.centroid.x, clipped_poly.centroid.y]) 
      else:
        centroid = np.array([np.nan, np.nan]) # If the polygon is empty append NaN
      
      return [clipped_poly], [centroid]
  
  
  def plot_cells(self, screen, world_to_pixel_func, polygons, centroids=None, cell_color=(255, 255, 204, 100)):
    """
    Draws the Voronoi cells on a given Pygame screen.

    Args:
        screen: Pygame surface.
        world_to_pixel_func: Function to convert world coords to pixels.
        polygons: list of shapely Polygons representing the Voronoi cells.
        centroids: optional np array of shape (n, 2) containing the centroids.
        cell_color: RGBA color tuple for the cell fill.
    """
    import pygame
    
    # Temporary surface for transparent polygons
    temp_surf = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)

    for poly in polygons:
        if poly is not None and not poly.is_empty:
            # Convert all polygon vertices to pixel coordinates
            px_vertices = [world_to_pixel_func(x, y) for x, y in zip(*poly.exterior.xy)]
            
            # Fill cell with alpha
            pygame.draw.polygon(temp_surf, cell_color, px_vertices)
            # Draw dashed-like border (Pygame doesn't support dashed natively, drawing solid thin line)
            pygame.draw.polygon(screen, (0, 0, 0), px_vertices, 1) 

    screen.blit(temp_surf, (0, 0))

    # Draw centroids as small red dots
    if centroids is not None:
        for centroid in centroids:
            if not np.isnan(centroid).any():
                cx, cy = world_to_pixel_func(centroid[0], centroid[1])
                pygame.draw.circle(screen, (0, 0, 0), (cx, cy), 4) # Red dot