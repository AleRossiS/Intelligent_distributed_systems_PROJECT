import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt

# Import custom modules
from robot_class import Robot
from environment_class import Environment
from voronoi_cells import VoronoiCell
from motion_and_mapping import MotionController
from broadcast_and_consensus import broadcast_data
from ekf_localization import GlobalCamera
import config as cfg

# --- 1. SETUP INIZIALE ---
pygame.init()

env = Environment(width=1200)
screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
pygame.display.set_caption("Valutazione Errore EKF e Occlusioni")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 12, bold=True)

# --- 2. CREAZIONE OSTACOLI ---
env.add_walls()
env.add_obstacle([(7, 1), (9, 1), (9, 3), (7, 3)])
env.add_obstacle([(-9, -2), (-5, -2), (-5, -1.5), (-9, -1.5)])
env.add_obstacle([(-5, -5.5), (-1, -5.5), (-1, -5), (-5, -5)])

cx1, cy1, raggio1, num_lati1 = -5.0, 6.0, 1.0, 20
cerchio1 = [(cx1 + raggio1 * np.cos(2 * np.pi * i / num_lati1), cy1 + raggio1 * np.sin(2 * np.pi * i / num_lati1)) for i in range(num_lati1)]
cx2, cy2, raggio2, num_lati2 = 7.0, -6.0, 1.0, 20
cerchio2 = [(cx2 + raggio2 * np.cos(2 * np.pi * i / num_lati2), cy2 + raggio2 * np.sin(2 * np.pi * i / num_lati2)) for i in range(num_lati2)]

env.add_obstacle(cerchio1)
env.add_obstacle(cerchio2)

treasure_pos = (11.5, -5)
env.add_target_object(cx=treasure_pos[0], cy=treasure_pos[1])

# --- 3. SETUP ENTI E CONTROLLORI ---
global_camera = GlobalCamera(x=14.0, y=9.0)

r1 = Robot(robot_id=1, x=-7.0, y=3.0, color='tab:red')
r2 = Robot(robot_id=2, x=-6.0, y=1.0, color='tab:blue')
r3 = Robot(robot_id=3, x=-8.5, y=2.5, color='tab:green') 

env.add_robot(r1)
env.add_robot(r2)
env.add_robot(r3)

vor_r1, vor_r2, vor_r3 = VoronoiCell(), VoronoiCell(), VoronoiCell()
ctrl1, ctrl2, ctrl3 = MotionController(), MotionController(), MotionController()

dt = 0.1

for current_robot in env.robots:
    for other_robot in env.robots:
        if current_robot.robot_id != other_robot.robot_id:
            current_robot.other_robots_positions[other_robot.robot_id] = (other_robot.state[0], other_robot.state[1])

# ==========================================
# --- DATI PER IL PLOT DELL'ERRORE EKF ---
# ==========================================
time_history = []
pos_error_history = []    # Distanza euclidea tra pos vera e stimata (Robot 1)
theta_error_history = []  # Differenza angolare (Robot 1)
cam_signal_history = []   # True se la camera vede il robot, False se lo perde

# --- 4. MAIN LOOP ---
step_counter = 0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    pos_r1 = np.vstack((r1.state[:2], list(r1.other_robots_positions.values())))
    pos_r2 = np.vstack((r2.state[:2], list(r2.other_robots_positions.values())))
    pos_r3 = np.vstack((r3.state[:2], list(r3.other_robots_positions.values())))

    real_obs_map = env.get_obstacle_polygons()
    inflated_obs_map = env.get_inflated_obstacles()

    cella1, centroid1 = vor_r1.compute_and_clip(r1.true_state, r1.state, pos_r1, robot_index=0, obstacles=inflated_obs_map)
    cella2, centroid2 = vor_r2.compute_and_clip(r2.true_state, r2.state, pos_r2, robot_index=0, obstacles=inflated_obs_map)
    cella3, centroid3 = vor_r3.compute_and_clip(r3.true_state, r3.state, pos_r3, robot_index=0, obstacles=inflated_obs_map)

    for r in [r1, r2, r3]:
        if not hasattr(r, 'simulation_time'):
            r.simulation_time = 0.0
        r.simulation_time += dt

    u1 = ctrl1.get_control(r1, dt, cella1[0], inflated_obs_map, env.target_object)
    u2 = ctrl2.get_control(r2, dt, cella2[0], inflated_obs_map, env.target_object)
    u3 = ctrl3.get_control(r3, dt, cella3[0], inflated_obs_map, env.target_object)

    for r in env.robots:
        if r.tick_timer():
            broadcast_data(r, env.robots)
            r.is_transmitting = True
        else:
            r.is_transmitting = False

    cam_statuses = {}
    for r, u in zip([r1, r2, r3], [u1, u2, u3]):
        z_camera = global_camera.get_measurement(r.true_state, real_obs_map)
        r.update_pose(u, dt, camera_measurement=z_camera)
        cam_statuses[r.robot_id] = "TRACKING" if z_camera is not None else "NO SIGNAL"

    # ==========================================
    # --- REGISTRAZIONE DATI (Robot 1) ---
    # ==========================================
    current_time = step_counter * dt
    time_history.append(current_time)
    
    # 1. Errore di Posizione (Distanza Euclidea)
    pos_err = np.hypot(r1.true_state[0] - r1.state[0], r1.true_state[1] - r1.state[1])
    pos_error_history.append(pos_err)
    
    # 2. Errore di Orientamento (Assoluto e normalizzato tra -pi e pi)
    theta_err = abs((r1.true_state[2] - r1.state[2] + np.pi) % (2 * np.pi) - np.pi)
    theta_error_history.append(theta_err)
    
    # 3. Stato del segnale della telecamera
    cam_signal_history.append(cam_statuses[1] == "TRACKING")

    # --- RENDERING ---
    cam_position = (global_camera.pos.x, global_camera.pos.y)
    env.render(screen, font, camera_statuses=cam_statuses, camera_pos=cam_position)
    
    vor_r1.plot_cells(screen, env.world_to_pixel, polygons=cella1, centroids=centroid1, cell_color=(255, 255, 204, 70))
    vor_r2.plot_cells(screen, env.world_to_pixel, polygons=cella2, centroids=centroid2, cell_color=(255, 255, 204, 70))
    vor_r3.plot_cells(screen, env.world_to_pixel, polygons=cella3, centroids=centroid3, cell_color=(255, 255, 204, 70))

    cam_px = env.world_to_pixel(global_camera.pos.x, global_camera.pos.y)
    pygame.draw.rect(screen, (255, 0, 255), (cam_px[0]-5, cam_px[1]-5, 10, 10))

    pygame.display.flip()
    clock.tick(30)
    
    step_counter += 1

pygame.quit()


# ==========================================
# --- PLOT DEGLI ERRORI EKF ---
# ==========================================
print("\nGenerazione dei plot dell'errore EKF in corso...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("EKF Estimation Error vs True State (Robot 1)", fontsize=16, fontweight='bold')

# Convertiamo in numpy array per usare le funzioni logiche di matplotlib
time_history = np.array(time_history)
pos_error_history = np.array(pos_error_history)
theta_error_history = np.array(theta_error_history)
cam_signal_history = np.array(cam_signal_history)
no_signal_mask = np.logical_not(cam_signal_history) # Aree in cui il segnale è perso

# --- SUBPLOT 1: Posizione ---
ax1.plot(time_history, pos_error_history, label='Euclidean Position Error', color='tab:red', linewidth=2)
# Evidenzia le fasce di perdita segnale (usiamo fill_between con l'asse X come riferimento)
ax1.fill_between(time_history, 0, 1, where=no_signal_mask, color='gray', alpha=0.3, transform=ax1.get_xaxis_transform(), label='No Camera Signal (Occlusion)')

ax1.set_ylabel("Position Error [m]", fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='upper right', fontsize=11)
ax1.set_ylim(bottom=0)
max_err = np.max(pos_error_history)
ax1.set_ylim(0, max_err * 1.5)

# --- SUBPLOT 2: Orientamento (Theta) ---
# Convertiamo in gradi per una lettura più intuitiva (opzionale, ma consigliato)
theta_error_deg = np.degrees(theta_error_history)

ax2.plot(time_history, theta_error_deg, label='Heading (Theta) Error', color='tab:blue', linewidth=2)
ax2.fill_between(time_history, 0, 1, where=no_signal_mask, color='gray', alpha=0.3, transform=ax2.get_xaxis_transform(), label='No Camera Signal (Occlusion)')

ax2.set_xlabel("Simulation Time [s]", fontsize=12)
ax2.set_ylabel("Heading Error [deg]", fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc='upper right', fontsize=11)
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.show()

sys.exit()