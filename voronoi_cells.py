import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import polygons
from shapely import geometry
import shapely
from shapely.geometry import Polygon, box, point, MultiPoint, LineString, Point
import Intelligent_distributed_systems_PROJECT.config as cfg
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

  def ray_casting_polygon(self, robot_pos, obstacles):
    """
    Compute the visibility polygon for a robot simulating ray casting.
    
    Args:
        robot_pos: tuple (x, y) representing the position of the robot.
        obstacles: list of shapely Polygons representing the obstacles in the environment.
    
    Returns:
        A shapely Polygon representing the visibility area of the robot.
    """
    ray_vertices = [] # Intialize list of vertices for the visibility polygon
    robot_point = Point(robot_pos) # Convert robot position to a shapely Point for distance calculations
        
    # Inflate obstacles 
    inflated_obstacles = [obs.buffer(cfg.OBS_INFLATION_RADIUS) for obs in obstacles]
    if inflated_obstacles:
      # Create a single polygon representing the union of all inflated obstacles for efficient difference operations
      collision_map = unary_union(inflated_obstacles).buffer(0) # Buffer(0) to clean up any potential geometry issue
    else:
      collision_map = Polygon() # Empty polygon if there are no obstacles
    
    # 1. Cast rays in all directions and find the first intersection with obstacles or environment boundaries
    for angle in self.angles:
      dx = np.cos(angle) * cfg.SENSING_RANGE
      dy = np.sin(angle) * cfg.SENSING_RANGE
      ray_end = (robot_pos[0] + dx, robot_pos[1] + dy)
      
      # Create a line representing the ray from the robot to the end point
      ray_line = LineString([robot_pos, ray_end]) 
      
      # Cut from the ray line the parts that intersect with the collision map 
      if not collision_map.is_empty:
        visible_ray = ray_line.difference(collision_map)
      else:
        visible_ray = ray_line # If there are no obstacles, the entire ray is visible
      
      # Initialize the hit point as the end of the ray (in case there are no intersections)
      hit_point = ray_end
      
      # If the visible part of the ray is not empty, we need to find the point where it ends 
      if not visible_ray.is_empty:
        # If the ray is split into multiple segments (e.g., it goes through an obstacle and comes out the other side)
        if visible_ray.geom_type == 'MultiLineString':
          # Check each segment of the visible ray and find the one that starts at the robot position (desired one)
          for line in visible_ray.geoms:
            # Check if the line starts at the robot position (very close to it)
            if line.distance(robot_point) < 1e-5:
              hit_point = line.coords[-1] # Coordinates of the end of the searched segment 
              break
        # If the ray is a single line, we can directly take the end point of that line
        elif visible_ray.geom_type == 'LineString':
          hit_point = visible_ray.coords[-1]
        
      # Add the hit point to the list of vertices for the visibility polygon
      ray_vertices.append(hit_point)
    
    # 2. Build the visibility polygon from the ray vertices
    # If the number of vertices is less than 3, we cannot form a polygon, so we return a small circle around the robot position as a fallback
    if len(ray_vertices) < 3:
      return robot_point.buffer(0.01)
      
    vis_poly = Polygon(ray_vertices) # Create a polygon from the ray vertices

    
    return vis_poly


  def compute_and_clip(self, points, robot_index, obstacles=[]):
    """Compute the Voronoi cell for the given robot and clip the cells to the sensing range and obstacles (including the environment boundaries) using a ray casting approach.

    Args:
        points: list of (x, y) tuples representing all robots positions.
        robot_index: index of the robot for which we want to compute the Voronoi cell.
        
        ex: points = [(1, 1), (3, 1), (2, 3)]
            robot_index = 0 (we want the Voronoi cell for the robot at (1, 1))

    Returns:
        shapely Polygon representing the clipped Voronoi cell for the specified robot.
        Centroid of the cell as a numpy array [cx, cy].
    """
    
    ### 1. COMPUTE VORONOI CELL ### 
    # Def a very large distance to handle infinite vertices of the cells
    far_away = max(self.bounds) * 100
    
    # Create dummy robot positons as a trick to ensure all Voronoi cells are finite
    # We postition these points at the corners of a large rectangle that encompasses the environment
    dummies = np.array([
        [self.bounds[0] - far_away, self.bounds[1] - far_away], # left bottom
        [self.bounds[2] + far_away, self.bounds[1] - far_away], # right bottom
        [self.bounds[0] - far_away, self.bounds[3] + far_away], # left top
        [self.bounds[2] + far_away, self.bounds[3] + far_away]  # right top
    ])
    
    # Combine the actual robot positions with the dummy points
    all_points = np.vstack((points, dummies))
    
    # Computes the Voronoi diagram for all the points (robots + dummies)
    vor = Voronoi(all_points) # From scipy library
    
    # Extract the Voronoi region corresponding to the specified robot index
    region_idx = vor.point_region[robot_index]
    region_vertices_indices = vor.regions[region_idx]
    
    #Extract the robot position
    robot_pos = points[robot_index]
    
    # Create a polygon from the vertices of the Voronoi region, if all vertices are finite (i.e., no -1 in the indices)
    if -1 not in region_vertices_indices and len(region_vertices_indices) > 0:
      
      # 1. Voronoi polygons computation   
      vor_poly = Polygon([vor.vertices[v] for v in region_vertices_indices])
      
      # Inflate the Voronoi cell by a safety margin )
      vor_poly = vor_poly.buffer(-cfg.DIST_BTW_ROBOTS)
        
      # 2. Ray Casting 
      vis_poly = self.ray_casting_polygon(robot_pos, obstacles)
      
      # 3. Clipping Voronoi cell with visibility polygon (obcles and environment boundaries)
      current_poly = vor_poly.intersection(vis_poly)
      
      # 4. Handles the case of multiple disjoint polygons (we had some errors otherwise)
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
      centroid = np.array([clipped_poly.centroid.x, clipped_poly.centroid.y]) # Extract the x and y coordinates of the centroid and append to the list
    else:
      centroid = np.array([np.nan, np.nan]) # If the polygon is empty append NaN
    
    
    return [clipped_poly], [centroid] # Return both the clipped polygon and its centroid
  
  
  def plot_cells(self, ax, polygons, centroids=None, cell_color='#ffffcc'):
      """Plot the Voronoi cells on a given Matplotlib axis.

      Args:
          ax: a Matplotlib axis object where the Voronoi cells will be drawn.
          polygons: list of shapely Polygons representing the Voronoi cells.
          centroids: optional np array of shape (n, 2) containing the centroids of the polygons.
          cell_color: color to use for drawing each Voronoi cell.
      """

      for poly in polygons:
        if poly is not None and not poly.is_empty:
          x, y = poly.exterior.xy
          
          ax.fill(x, y, alpha=0.5, color=cell_color, zorder=0)
          ax.plot(x, y, color='black', linewidth=1, linestyle='--', zorder=0)

      if centroids is not None:
        for cx, cy in centroids:
          if not np.isnan(cx) and not np.isnan(cy):
            ax.plot(cx, cy, marker='x', color='black', markersize=8, zorder=1)