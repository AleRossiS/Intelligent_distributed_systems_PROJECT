import pygame
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import config as cfg
import numpy as np
import math

class Environment:
    """
    Class representing a 2D environment for robot simulation, including obstacles, boundaries, 
    and Pygame rendering capabilities.
    """
    def __init__(self, width=1200):
        self.xlim = cfg.X_LIM
        self.ylim = cfg.Y_LIM
        self.robots = []     # List of Robot instances in the environment
        self.obstacles = []  # List of obstacles vertices (each obstacle is defined as a list of (x, y) tuples)
        self.pickup_zone = None
        self.delivery_zone = None
        self.target_object = None
        
        # --- Pygame Screen Setup ---
        self.span_x = self.xlim[1] - self.xlim[0]
        self.span_y = self.ylim[1] - self.ylim[0]
        self.WIDTH = width
        self.HEIGHT = int(self.WIDTH * (self.span_y / self.span_x))
        
        # Standard RGB colors for Pygame
        self.COLORS = {
            'gray': (128, 128, 128),
            'gold': (255, 215, 0),
            'darkorange': (255, 140, 0),
            'cyan': (0, 255, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'magenta': (255, 0, 255)
        }

    def add_robot(self, robot):
        """Adds a robot to the environment.
        args:
            robot: an instance of the Robot class to be added to the environment.
        """
        self.robots.append(robot)
        
    def add_obstacle(self, vertices):
        """
        Adds a generic polygonal obstacle and returns the updated list of obstacles.
        Args:
            vertices: list of (x, y) tuples representing the polygon vertices.
                      e.g., [(0,0), (2,0), (2,2), (0,2)]
        """
        self.obstacles.append(vertices)
        
    def get_obstacle_polygons(self):
        """Returns a single merged Shapely geometry representing all obstacles."""
        polygons = [Polygon(obs) for obs in self.obstacles]
        
        # Add the target object as an obstacle if it exists
        if self.target_object is not None:
            polygons.append(self.target_object)
        
        # Merge all obstacles into a single geometry to simplify intersection checks
        merged_obstacles = unary_union(polygons)
        return merged_obstacles

    def get_inflated_obstacles(self):
        """Returns the inflated obstacle map."""
        real_map = self.get_obstacle_polygons()
        return real_map.buffer(cfg.OBS_INFLATION_RADIUS)

    def add_walls(self, wall_thickness=0.5):
        """Adds walls around the environment boundaries as obstacles."""
        min_x, max_x = self.xlim
        min_y, max_y = self.ylim
        
        # Bottom wall
        self.add_obstacle([(min_x, min_y + wall_thickness), (max_x, min_y + wall_thickness), 
                           (max_x, min_y), (min_x, min_y)])
        # Top wall
        self.add_obstacle([(min_x, max_y - wall_thickness), (max_x, max_y - wall_thickness), 
                           (max_x, max_y), (min_x, max_y)])
        # Left wall
        self.add_obstacle([(min_x + wall_thickness, min_y), (min_x, min_y), 
                           (min_x, max_y), (min_x + wall_thickness, max_y)])
        # Right wall
        self.add_obstacle([(max_x, min_y), (max_x - wall_thickness, min_y), 
                           (max_x - wall_thickness, max_y), (max_x, max_y)])

    def add_target_object(self, cx, cy, radius=0.4):
        """
        Adds a target object (circle shape) as a Shapely Polygon to be detectable by LiDAR.
        Args:
            cx: x coordinate of the circle center.
            cy: y coordinate of the circle center.
            radius: radius of the circular target object.
        """
        self.target_object = Point(cx, cy).buffer(radius)

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
        
        # Create a circle for the signal range and subtract the obstacles
        signal_circle = tx_point.buffer(sig_range)
        obs_map = self.get_obstacle_polygons()
        
        if obs_map.is_empty:
            return signal_circle
            
        return signal_circle.difference(obs_map)


    def world_to_pixel(self, x, y):
        """Converts world coordinates (meters) to Pygame pixel coordinates."""
        px = int((x - self.xlim[0]) / self.span_x * self.WIDTH)
        # Invert Y axis for Pygame (0 is top)
        py = int(self.HEIGHT - ((y - self.ylim[0]) / self.span_y * self.HEIGHT))
        return px, py

    def render(self, screen, font=None, camera_statuses=None, camera_pos=None):
        """
        Main draw call. Renders the environment, obstacles, target, and all robots.
        Args:
            screen: Pygame surface.
            font: Pygame font object for text overlay.
            camera_statuses: dict mapping robot_id to string status (optional).
            camera_pos: tuple representing the position of the camera (optional).
        """
        screen.fill(self.COLORS['white']) # Clear screen

        # Draw obstacles
        for obs_vertices in self.obstacles:
            px_vertices = [self.world_to_pixel(v[0], v[1]) for v in obs_vertices]
            pygame.draw.polygon(screen, self.COLORS['gray'], px_vertices)
            pygame.draw.polygon(screen, self.COLORS['black'], px_vertices, 2) # Outline

        # Draw treasure signal area (with transparency)
        sig_area = self.get_treasure_signal_polygon()
        if sig_area and not sig_area.is_empty and sig_area.geom_type == 'Polygon':
            sig_coords = list(sig_area.exterior.coords)
            px_sig = [self.world_to_pixel(c[0], c[1]) for c in sig_coords]
            
            # Temporary surface for alpha channel
            temp_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(temp_surface, (0, 255, 255, 50), px_sig) # Cyan fill
            pygame.draw.polygon(temp_surface, (0, 191, 255, 150), px_sig, 2) # Border
            screen.blit(temp_surface, (0, 0))

        # Draw target object
        if self.target_object is not None:
            target_coords = list(self.target_object.exterior.coords)
            px_target = [self.world_to_pixel(c[0], c[1]) for c in target_coords]
            pygame.draw.polygon(screen, self.COLORS['gold'], px_target)
            pygame.draw.polygon(screen, self.COLORS['darkorange'], px_target, 2)

        # Draw robots
        for r in self.robots:
            
            cam_stat = camera_statuses.get(r.robot_id) if camera_statuses else None
            r.draw(screen, self.world_to_pixel, font, cam_stat, camera_pos)