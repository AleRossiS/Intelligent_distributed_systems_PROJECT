import numpy as np
import matplotlib.pyplot as plt

import config as cfg

class Robot:
  """
  A simple unicycle robot class for 2D simulation
  """ 
  
  def __init__(self, robot_id, x = 0.0, y = 0.0, theta = 0.0, radius = cfg.ROBOT_RADIUS, color = 'blue'):
    self.robot_id = robot_id
    self.state = np.array([x, y, theta], dtype=float) # [x, y, theta]
    self.radius = radius
    self.color = color
    self.history_x = [self.state[0]]
    self.history_y = [self.state[1]]
    
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
                        alpha=0.5)
    ax.add_patch(circle)
    
    front_x = x + self.radius * np.cos(theta)
    front_y = y + self.radius * np.sin(theta)
    ax.plot([x, front_x], [y, front_y], color='black', linewidth=2)
    
    # Label the robot with its ID
    ax.text(x, y + self.radius*1.2, f"R{self.robot_id}", ha='center')
    
    # Plot robto trajectory
    if show_history:
        ax.plot(self.history_x, self.history_y, color=self.color, linewidth=1.5, alpha=0.4, linestyle='-')
    
  