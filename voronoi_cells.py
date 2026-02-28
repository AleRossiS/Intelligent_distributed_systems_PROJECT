import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import polygons
from shapely.geometry import Polygon, box


class VoronoiCell:
  """
  Class that handles the Voronoi cell of a robot, 
  given the positions of all robots and the environment boundaries.
  
  Attributes:
    bounds: list of the form [x_min, y_min, x_max, y_max] defining the environment boundaries.
    boundary_poly: shapely Polygon representing the environment boundaries.
  """
  
  def __init__(self, x_lim, y_lim):
    self.bounds = [x_lim[0], y_lim[0], x_lim[1], y_lim[1]]
    self.boundary_poly = box(*self.bounds) # Create a rectangle for the environment boundaries (using shapely library)


  def compute_and_clip(self, points):
    """Compute the Voronoi diagram for the given points and clip the cells to the environment boundaries.

    Args:
        points: list of (x, y) tuples representing robot positions.

    Returns:
        list of shapely Polygons representing the clipped (with the environment boundaries) Voronoi cells.
    """
    
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
    clipped_polygons = [] # Initialize list to store the clipped Voronoi cells

    
    # Loop through each Voronoi region (loop through the original robot points only)
    for i in range(len(points)):
      region_idx = vor.point_region[i] # Get the index of the Voronoi region corresponding to the i-th robot
      region = vor.regions[region_idx] # Get the vertices idx of that cell
      
      # Create a polygon for the Voronoi cell using the vertices coordinates
      poly = Polygon([vor.vertices[v] for v in region])
      
      # Clip the Voronoi cell polygon to the environment boundaries using shapely's intersection method
      clipped_poly = poly.intersection(self.boundary_poly)
      clipped_polygons.append(clipped_poly)
        
    return clipped_polygons
  
  
  def get_centroids(self, polygons):
    """Calculate the centroids of a list of shapely Polygons.

    Args:
        polygons: list of shapely Polygons.

    Returns:
        np array of shape (n, 2) containing the centroids of the polygons, where n is the number of polygons.
    """
    # Initialize an empty list to store the centroids
    centroids = []
    
    # Loop through each polygon (polygons is a list of shapely Polygons)
    for poly in polygons:
      if poly is not None and not poly.is_empty:
        centroids.append([poly.centroid.x, poly.centroid.y]) # Extract the x and y coordinates of the centroid and append to the list
      else:
        centroids.append([np.nan, np.nan]) # If the polygon is empty append NaN
        
    return np.array(centroids)
  
  
  def plot_cells(self, ax, polygons, centroids=None, cell_color='#ffffcc'):
      """Plot the Voronoi cells on a given Matplotlib axis.

      Args:
          ax: a Matplotlib axis object where the cells will be drawn.
          polygons: list of shapely Polygons representing the Voronoi cells.
          centroids: optional np array of shape (n, 2) containing the centroids of the polygons.
          cell_color: color to use for drawing each Voronoi cell.
      """

      for poly in polygons:
        if poly is not None and not poly.is_empty:
          x, y = poly.exterior.xy
          
          ax.fill(x, y, alpha=0.5, color=cell_color, zorder=0)
          ax.plot(x, y, color='black', linewidth=1.5, linestyle='--', zorder=0)

      if centroids is not None:
        for cx, cy in centroids:
          if not np.isnan(cx) and not np.isnan(cy):
            ax.plot(cx, cy, marker='x', color='black', markersize=8, zorder=1)