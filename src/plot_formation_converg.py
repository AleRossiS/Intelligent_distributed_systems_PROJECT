import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt

from robot_class import Robot
from environment_class import Environment
from voronoi_cells import VoronoiCell
from motion_and_mapping import MotionController
from broadcast_and_consensus import broadcast_data
from ekf_localization import GlobalCamera
import config as cfg

# --- PARAMETRI MULTI-RUN ---
NUM_RUNS = 4
MAX_STEPS_PER_RUN = 10000          # tetto di sicurezza (~16 min), il vero stop è il timer
dt = 0.1
POST_DISCOVERY_SECONDS = 30.0
POST_DISCOVERY_STEPS   = int(POST_DISCOVERY_SECONDS / dt)   # 300 step

CONVERGENCE_THRESHOLD = 1.0        # metri — i 30s partono quando TUTTI sono sotto soglia

target_ring_radius = getattr(cfg, 'RING_RADIUS', 1.5)

all_runs_data = []

pygame.init()

for run_idx in range(NUM_RUNS):
    print(f"\n================================")
    print(f"   STARTING RUN {run_idx + 1}/{NUM_RUNS}")
    print(f"================================")

    # --- 1. SETUP INIZIALE ---
    env = Environment(width=1200)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(f"Simulazione - RUN {run_idx + 1}")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont('Arial', 12, bold=True)

    # --- 2. OSTACOLI E TESORO ---
    env.add_walls()
    env.add_obstacle([(7, 1), (9, 1), (9, 3), (7, 3)])
    env.add_obstacle([(-9, -2), (-5, -2), (-5, -1.5), (-9, -1.5)])
    env.add_obstacle([(-5, -5.5), (-1, -5.5), (-1, -5), (-5, -5)])

    cx1, cy1, raggio1, num_lati1 = -5.0, 6.0, 1.0, 20
    cerchio1 = [(cx1 + raggio1 * np.cos(2 * np.pi * i / num_lati1),
                 cy1 + raggio1 * np.sin(2 * np.pi * i / num_lati1)) for i in range(num_lati1)]
    cx2, cy2, raggio2, num_lati2 = 7.0, -6.0, 1.0, 20
    cerchio2 = [(cx2 + raggio2 * np.cos(2 * np.pi * i / num_lati2),
                 cy2 + raggio2 * np.sin(2 * np.pi * i / num_lati2)) for i in range(num_lati2)]
    env.add_obstacle(cerchio1)
    env.add_obstacle(cerchio2)

    treasure_pos = (11.5, -5)
    env.add_target_object(cx=treasure_pos[0], cy=treasure_pos[1])

    # --- 3. ROBOT E CONTROLLORI ---
    global_camera = GlobalCamera(x=14.0, y=9.0)

    noise = np.random.uniform(-2.0, 2.0, 6)
    r1 = Robot(robot_id=1, x=-7.0 + noise[0], y=3.0 + noise[1], color='tab:red')
    r2 = Robot(robot_id=2, x=-6.0 + noise[2], y=1.0 + noise[3], color='tab:blue')
    r3 = Robot(robot_id=3, x=-8.5 + noise[4], y=2.5 + noise[5], color='tab:green')

    env.add_robot(r1)
    env.add_robot(r2)
    env.add_robot(r3)

    vor_r1, vor_r2, vor_r3 = VoronoiCell(), VoronoiCell(), VoronoiCell()
    ctrl1,  ctrl2,  ctrl3  = MotionController(), MotionController(), MotionController()

    for current_robot in env.robots:
        for other_robot in env.robots:
            if current_robot.robot_id != other_robot.robot_id:
                current_robot.other_robots_positions[other_robot.robot_id] = (
                    other_robot.state[0], other_robot.state[1])

    # --- DATI PLOT ---
    time_history       = []
    ring_error_history = {1: [], 2: [], 3: []}
    time_treasure_found = None     # primo robot che trova il tesoro (marker sul plot)

    # --- TIMER POST-CONVERGENZA ---
    all_found_step  = None         # step in cui TUTTI scendono sotto soglia
    steps_after_all = 0

    # --- 4. MAIN LOOP ---
    step_counter = 0
    running = True

    while running and step_counter < MAX_STEPS_PER_RUN:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Voronoi ---
        pos_r1 = np.vstack((r1.state[:2], list(r1.other_robots_positions.values())))
        pos_r2 = np.vstack((r2.state[:2], list(r2.other_robots_positions.values())))
        pos_r3 = np.vstack((r3.state[:2], list(r3.other_robots_positions.values())))

        real_obs_map     = env.get_obstacle_polygons()
        inflated_obs_map = env.get_inflated_obstacles()

        cella1, centroid1 = vor_r1.compute_and_clip(r1.true_state, r1.state, pos_r1, robot_index=0, obstacles=inflated_obs_map)
        cella2, centroid2 = vor_r2.compute_and_clip(r2.true_state, r2.state, pos_r2, robot_index=0, obstacles=inflated_obs_map)
        cella3, centroid3 = vor_r3.compute_and_clip(r3.true_state, r3.state, pos_r3, robot_index=0, obstacles=inflated_obs_map)

        for r in [r1, r2, r3]:
            if not hasattr(r, 'simulation_time'):
                r.simulation_time = 0.0
            r.simulation_time += dt

        # --- Controllo ---
        u1 = ctrl1.get_control(r1, dt, cella1[0], inflated_obs_map, env.target_object)
        u2 = ctrl2.get_control(r2, dt, cella2[0], inflated_obs_map, env.target_object)
        u3 = ctrl3.get_control(r3, dt, cella3[0], inflated_obs_map, env.target_object)

        # --- Rete ---
        for r in env.robots:
            if r.tick_timer():
                broadcast_data(r, env.robots)
                r.is_transmitting = True
            else:
                r.is_transmitting = False

        # --- Fisica + EKF ---
        cam_statuses = {}
        for r, u in zip([r1, r2, r3], [u1, u2, u3]):
            z_camera = global_camera.get_measurement(r.true_state, real_obs_map)
            r.update_pose(u, dt, camera_measurement=z_camera)
            cam_statuses[r.robot_id] = "TRACKING" if z_camera is not None else "NO SIGNAL"

        # --- Rendering ---
        cam_position = (global_camera.pos.x, global_camera.pos.y)
        env.render(screen, font, camera_statuses=cam_statuses, camera_pos=cam_position)

        vor_r1.plot_cells(screen, env.world_to_pixel, polygons=cella1, centroids=centroid1, cell_color=(255, 255, 204, 70))
        vor_r2.plot_cells(screen, env.world_to_pixel, polygons=cella2, centroids=centroid2, cell_color=(255, 255, 204, 70))
        vor_r3.plot_cells(screen, env.world_to_pixel, polygons=cella3, centroids=centroid3, cell_color=(255, 255, 204, 70))

        cam_px = env.world_to_pixel(global_camera.pos.x, global_camera.pos.y)
        pygame.draw.rect(screen, (255, 0, 255), (cam_px[0]-5, cam_px[1]-5, 10, 10))

        # Titolo finestra con feedback
        if all_found_step is not None:
            remaining = POST_DISCOVERY_SECONDS - (steps_after_all * dt)
            pygame.display.set_caption(f"RUN {run_idx+1} — Converged! Stopping in {remaining:.0f}s")
        else:
            pygame.display.set_caption(f"RUN {run_idx+1} — Searching... t={step_counter * dt:.1f}s")

        pygame.display.flip()
        clock.tick(60)

        # --- RACCOLTA DATI ---
        current_time = step_counter * dt
        time_history.append(current_time)

        # Marker: primo robot che trova il tesoro
        if time_treasure_found is None:
            for r in env.robots:
                if getattr(r, 'state_mode', '') == "TREASURE_FOUND" or getattr(r, 'has_sensed_treasure', False):
                    time_treasure_found = current_time
                    break

        for r in env.robots:
            dist       = np.hypot(r.true_state[0] - treasure_pos[0],
                                  r.true_state[1] - treasure_pos[1])
            ring_error = abs(dist - target_ring_radius)
            ring_error_history[r.robot_id].append(ring_error)

        # -------------------------------------------------------
        # LOGICA DI STOP: 30s dopo che TUTTI i robot sono sotto soglia
        # -------------------------------------------------------
        all_converged = all(
            ring_error_history[r.robot_id][-1] < CONVERGENCE_THRESHOLD
            for r in env.robots
        )

        if all_converged:
            if all_found_step is None:
                all_found_step = step_counter
                print(f"  ★ Tutti i robot sotto soglia ({CONVERGENCE_THRESHOLD}m) "
                      f"al passo {step_counter} (t={current_time:.1f}s). "
                      f"Stop tra {POST_DISCOVERY_SECONDS}s.")

            steps_after_all += 1
            if steps_after_all >= POST_DISCOVERY_STEPS:
                print(f"  ✓ {POST_DISCOVERY_SECONDS}s trascorsi dopo convergenza. "
                      f"Fine run {run_idx + 1}.")
                running = False
        else:
            # Se l'errore risale sopra soglia il timer si resetta (comportamento one-shot: rimuovi questo else)
            if all_found_step is not None:
                print(f"  ⚠ Errore risalito sopra soglia al passo {step_counter}, timer resettato.")
            all_found_step  = None
            steps_after_all = 0

        step_counter += 1

    all_runs_data.append({
        'time':       time_history,
        'errors':     ring_error_history,
        'found_time': time_treasure_found
    })

pygame.quit()


# ==========================================
# --- PLOT AGGREGATO ---
# ==========================================
print("\nGenerazione del plot aggregato in corso...")

plt.figure(figsize=(12, 7))

for idx, data in enumerate(all_runs_data):
    t    = data['time']
    errs = data['errors']
    tf   = data['found_time']

    label_r1 = 'Robot 1' if idx == 0 else None
    label_r2 = 'Robot 2' if idx == 0 else None
    label_r3 = 'Robot 3' if idx == 0 else None

    plt.plot(t, errs[1], label=label_r1, color='tab:red',   linewidth=1.5, alpha=0.6)
    plt.plot(t, errs[2], label=label_r2, color='tab:blue',  linewidth=1.5, alpha=0.6)
    plt.plot(t, errs[3], label=label_r3, color='tab:green', linewidth=1.5, alpha=0.6)

    if tf is not None:
        label_tf = 'Treasure Discovered (first)' if idx == 0 else None
        plt.axvline(x=tf, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label=label_tf)

plt.title(f"Swarm Convergence to Target Ring Formation ({NUM_RUNS} Runs Combined)",
          fontsize=16, fontweight='bold')
plt.xlabel("Simulation Time [s]", fontsize=12)
plt.ylabel(f"Distance Error from Target Ring (r={target_ring_radius}m) [m]", fontsize=12)
plt.ylim(bottom=0)
plt.axhline(y=CONVERGENCE_THRESHOLD, color='gray', linestyle=':', linewidth=1.5,
            label=f'Convergence threshold ({CONVERGENCE_THRESHOLD}m)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12, loc="upper right")
plt.tight_layout()
plt.show()

sys.exit()