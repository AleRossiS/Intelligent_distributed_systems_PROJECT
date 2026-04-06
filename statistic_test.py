import numpy as np
import pygame
import sys
import pickle

# Import custom modules
from robot_class import Robot
from environment_class import Environment
from voronoi_cells import VoronoiCell
from motion_and_mapping import MotionController
from broadcast_and_consensus import broadcast_data
from ekf_localization import GlobalCamera
import config as cfg

# cinfigurazione per i test statistici
HEADLESS_MODE = True
NUM_RUNS = 50
MAX_STEPS = 3000

experiment_data = {
    'runs_error': [],
    'arrival_times': []
}

# Solo se vogliamo vedere la grafica inizializziamo Pygame una volta sola
if not HEADLESS_MODE:
    pygame.init()
    # Usiamo un env temporaneo solo per farci dare WIDTH e HEIGHT
    temp_env = Environment(width=1200)
    screen = pygame.display.set_mode((temp_env.WIDTH, temp_env.HEIGHT))
    pygame.display.set_caption("Simulazione Robotica con Voronoi e EKF")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 12, bold=True)

for run in range(NUM_RUNS):
    print(f"Starting run {run+1}/{NUM_RUNS}...")

    env = Environment(width=1200)
    run_t1 = None
    run_t2 = None
    run_t3 = None

    # --- 2. CREAZIONE OSTACOLI (Dal tuo test vecchio) ---
    env.add_walls()
    env.add_obstacle([(7, 1), (9, 1), (9, 3), (7, 3)])
    env.add_obstacle([(-9, -2), (-5, -2), (-5, -1.5), (-9, -1.5)])
    env.add_obstacle([(-5, -5.5), (-1, -5.5), (-1, -5), (-5, -5)])

    # Il piccolo cerchio 1
    cx1, cy1 = -5.0, 7.0
    raggio1 = 1.0
    num_lati1 = 20
    cerchio1 = [(cx1 + raggio1 * np.cos(2 * np.pi * i / num_lati1), cy1 + raggio1 * np.sin(2 * np.pi * i / num_lati1)) for i in range(num_lati1)]

    # Il piccolo cerchio 2
    cx2, cy2 = 7.0, -6.0
    raggio2 = 1.0
    num_lati2 = 20
    cerchio2 = [(cx2 + raggio2 * np.cos(2 * np.pi * i / num_lati2), cy2 + raggio2 * np.sin(2 * np.pi * i / num_lati2)) for i in range(num_lati2)]

    env.add_obstacle(cerchio1)
    env.add_obstacle(cerchio2)

    # Tesoro
    treasure_pos = (12, -7)
    env.add_target_object(cx=treasure_pos[0], cy=treasure_pos[1])
    true_treasure_center = np.array([treasure_pos[0], treasure_pos[1]])

    # --- 3. SETUP ENTI E CONTROLLORI ---
    global_camera = GlobalCamera(x=14.0, y=9.0)

    r1 = Robot(robot_id=1, x=-1.0, y=3.0, color='tab:red')
    r2 = Robot(robot_id=2, x=0.0, y=1.0, color='tab:blue')
    r3 = Robot(robot_id=3, x=-7.5, y=-7.5, color='tab:green') 

    env.add_robot(r1)
    env.add_robot(r2)
    env.add_robot(r3)

    vor_r1 = VoronoiCell()
    vor_r2 = VoronoiCell()
    vor_r3 = VoronoiCell()

    ctrl1 = MotionController()
    ctrl2 = MotionController()
    ctrl3 = MotionController()

    dt = 0.1

    # INIZIALIZZAZIONE MEMORIA
    for current_robot in env.robots:
        for other_robot in env.robots:
            if current_robot.robot_id != other_robot.robot_id:
                current_robot.other_robots_positions[other_robot.robot_id] = (other_robot.state[0], other_robot.state[1])


    # --- 4. MAIN LOOP ---
    step_counter = 0
    run_error_history = []
    run_ttfd = None

    #main loop for run

    while step_counter < MAX_STEPS:

        if not HEADLESS_MODE:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
        
        # Preparazione array di posizioni stimate con EKF per il calcolo delle celle di Voronoi
        pos_r1 = np.vstack((r1.state[:2], list(r1.other_robots_positions.values())))
        pos_r2 = np.vstack((r2.state[:2], list(r2.other_robots_positions.values())))
        pos_r3 = np.vstack((r3.state[:2], list(r3.other_robots_positions.values())))

        real_obs_map = env.get_obstacle_polygons()
        inflated_obs_map = env.get_inflated_obstacles()

        # Voronoi compute_and_clip
        cella1, centroid1 = vor_r1.compute_and_clip(
            r1.true_state, r1.state, pos_r1, robot_index=0, obstacles=inflated_obs_map
        )
        
        cella2, centroid2 = vor_r2.compute_and_clip(
            r2.true_state, r2.state, pos_r2, robot_index=0, obstacles=inflated_obs_map
        )
        
        cella3, centroid3 = vor_r3.compute_and_clip(
            r3.true_state, r3.state, pos_r3, robot_index=0, obstacles=inflated_obs_map
        )

        for r in [r1, r2, r3]:
            if not hasattr(r, 'simulation_time'):
                r.simulation_time = 0.0
            r.simulation_time += dt

        # Controllo del Moto
        u1 = ctrl1.get_control(r1, env.robots, dt, cella1[0], real_obs_map, inflated_obs_map, env.target_object)
        u2 = ctrl2.get_control(r2, env.robots, dt, cella2[0], real_obs_map, inflated_obs_map, env.target_object)
        u3 = ctrl3.get_control(r3, env.robots, dt, cella3[0], real_obs_map, inflated_obs_map, env.target_object)

        # Rete e Trasmissione
        for r in env.robots:
            if r.tick_timer():
                broadcast_data(r, env.robots)
            #     r.is_transmitting = True
            # else:
            #     r.is_transmitting = False

        # Fisica e EKF + Preparazione per il Rendering
        cam_statuses = {}
        for r, u in zip([r1, r2, r3], [u1, u2, u3]):
            z_camera = global_camera.get_measurement(r.true_state, real_obs_map)
            r.update_pose(u, dt, camera_measurement=z_camera)
            
            # Registriamo lo stato per passarlo al renderer
            cam_statuses[r.robot_id] = "TRACKING" if z_camera is not None else "NO SIGNAL"
        
        # Raccolta dati statistici
        # Quanti robot hanno fisicamente raggiunto il tesoro e fatto la loro misurazione?
        sensed_count = sum([1 for r in [r1, r2, r3] if getattr(r, 'has_sensed_treasure', False)])
        
        # Registriamo i tempi di arrivo progressivi
        if sensed_count >= 1 and run_t1 is None: run_t1 = step_counter
        if sensed_count >= 2 and run_t2 is None: run_t2 = step_counter
        if sensed_count >= 3 and run_t3 is None: run_t3 = step_counter

        estimates = [getattr(r, 'consensus_treasure_pos', None) for r in [r1, r2, r3]]
        valid_estimates = [e for e in estimates if e is not None]
        
        if len(valid_estimates) > 0:
            swarm_avg = np.mean(valid_estimates, axis=0)
            error = np.linalg.norm(swarm_avg - true_treasure_center)
            run_error_history.append(error)
        else:
            run_error_history.append(np.nan)

        # ==========================================
        # --- 5. RENDERING PULITO CON LE CLASSI solo se headless = false ---
        # ==========================================
        
        if not HEADLESS_MODE:
            # 1. Disegna tutto l'ambiente, gli ostacoli, il tesoro e i robot
            cam_position = (global_camera.pos.x, global_camera.pos.y)
            env.render(screen, font, camera_statuses=cam_statuses, camera_pos=cam_position)
            
            # 2. Disegna le celle di Voronoi trasparenti per ogni robot
            vor_r1.plot_cells(screen, env.world_to_pixel, polygons=cella1, centroids=centroid1, cell_color=(255, 255, 204, 70))
            vor_r2.plot_cells(screen, env.world_to_pixel, polygons=cella2, centroids=centroid2, cell_color=(255, 255, 204, 70))
            vor_r3.plot_cells(screen, env.world_to_pixel, polygons=cella3, centroids=centroid3, cell_color=(255, 255, 204, 70))
            # Aggiorna lo schermo a 60 FPS
            pygame.display.flip()
            clock.tick(30)
        
        step_counter += 1

    experiment_data['runs_error'].append(run_error_history)
    experiment_data['arrival_times'].append((
        run_t1 if run_t1 is not None else MAX_STEPS,
        run_t2 if run_t2 is not None else MAX_STEPS,
        run_t3 if run_t3 is not None else MAX_STEPS
    ))

with open('experiment_results3.pkl', 'wb') as f:
    pickle.dump(experiment_data, f)

print("\nRun completed")
print("Data saved to 'experiment_results3.pkl'")