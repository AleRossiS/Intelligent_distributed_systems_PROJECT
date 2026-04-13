import numpy as np
import sys
import pickle
import concurrent.futures
import multiprocessing
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from time import time


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
NUM_RUNS = 100
MAX_STEPS = 6000

# =================================================================
# FUNZIONE DELLA SINGOLA SIMULAZIONE (Eseguita dai vari Core)
# =================================================================
def run_single_simulation(run_id):
    """Questa funzione esegue una simulazione completa in isolamento e restituisce i dati."""
    # (Non stampiamo nulla qui dentro per non sporcare il terminale)
    
    env = Environment(width=1200)
    env.add_walls()
    env.add_obstacle([(2, 0), (4, 0), (4, 2), (2, 2)])
    env.add_obstacle([(-9, -2), (-5, -2), (-5, -1.5), (-9, -1.5)])
    env.add_obstacle([(-5, -5.5), (-1, -5.5), (-1, -5), (-5, -5)])

    cerchio1 = [( -5.0 + 1.0 * np.cos(2 * np.pi * i / 20),  6.0 + 1.0 * np.sin(2 * np.pi * i / 20)) for i in range(20)]
    cerchio2 = [(  7.0 + 1.0 * np.cos(2 * np.pi * i / 20), -6.0 + 1.0 * np.sin(2 * np.pi * i / 20)) for i in range(20)]
    env.add_obstacle(cerchio1)
    env.add_obstacle(cerchio2)

    treasure_pos = (9, 4)
    env.add_target_object(cx=treasure_pos[0], cy=treasure_pos[1])
    true_treasure_center = np.array([treasure_pos[0], treasure_pos[1]])

    global_camera = GlobalCamera(x=14.0, y=9.0)
    r1 = Robot(robot_id=1, x=-1.0, y=3.0, color='tab:red')
    r2 = Robot(robot_id=2, x=0.0, y=1.0, color='tab:blue')
    r3 = Robot(robot_id=3, x=-7.5, y=-7.5, color='tab:green') 

    env.add_robot(r1); env.add_robot(r2); env.add_robot(r3)

    vor_r1, vor_r2, vor_r3 = VoronoiCell(), VoronoiCell(), VoronoiCell()
    ctrl1, ctrl2, ctrl3 = MotionController(), MotionController(), MotionController()
    dt = 0.1

    for current_robot in env.robots:
        for other_robot in env.robots:
            if current_robot.robot_id != other_robot.robot_id:
                current_robot.other_robots_positions[other_robot.robot_id] = (other_robot.state[0], other_robot.state[1])

    step_counter = 0
    run_error_history = []
    run_t1 = run_t2 = run_t3 = None

    while step_counter < MAX_STEPS:
        
        pos_r1 = np.vstack((r1.state[:2], list(r1.other_robots_positions.values())))
        pos_r2 = np.vstack((r2.state[:2], list(r2.other_robots_positions.values())))
        pos_r3 = np.vstack((r3.state[:2], list(r3.other_robots_positions.values())))

        real_obs_map = env.get_obstacle_polygons()
        inflated_obs_map = env.get_inflated_obstacles()

        cella1, _ = vor_r1.compute_and_clip(r1.true_state, r1.state, pos_r1, robot_index=0, obstacles=inflated_obs_map)
        cella2, _ = vor_r2.compute_and_clip(r2.true_state, r2.state, pos_r2, robot_index=0, obstacles=inflated_obs_map)
        cella3, _ = vor_r3.compute_and_clip(r3.true_state, r3.state, pos_r3, robot_index=0, obstacles=inflated_obs_map)

        for r in [r1, r2, r3]:
            if not hasattr(r, 'simulation_time'): r.simulation_time = 0.0
            r.simulation_time += dt

        u1 = ctrl1.get_control(r1, dt, cella1[0], inflated_obs_map, env.target_object)
        u2 = ctrl2.get_control(r2, dt, cella2[0], inflated_obs_map, env.target_object)
        u3 = ctrl3.get_control(r3, dt, cella3[0], inflated_obs_map, env.target_object)

        for r in env.robots:
            if r.tick_timer(): broadcast_data(r, env.robots)

        for r, u in zip([r1, r2, r3], [u1, u2, u3]):
            z_camera = global_camera.get_measurement(r.true_state, real_obs_map)
            r.update_pose(u, dt, camera_measurement=z_camera)

        # RACCOLTA DATI
        sensed_count = sum([1 for r in [r1, r2, r3] if getattr(r, 'has_sensed_treasure', False)])
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
            
        step_counter += 1

    arrival_times = (
        run_t1 if run_t1 else MAX_STEPS,
        run_t2 if run_t2 else MAX_STEPS,
        run_t3 if run_t3 else MAX_STEPS
    )

    # Estraiamo le traiettorie dell'intera run
    run_trajectories = []
    for r in [r1, r2, r3]:
        run_trajectories.append({
            'id': r.robot_id,
            'color': r.color,
            'x': r.history_true_x,
            'y': r.history_true_y
        })
    
    # Ritorna i dati al processo principale
    return run_error_history, arrival_times, run_trajectories


# =================================================================
# GESTORE MULTIPROCESSING E BARRA DI CARICAMENTO (Main Process)
# =================================================================
if __name__ == '__main__': # <-- FONDAMENTALE PER IL MULTIPROCESSING IN PYTHON

    multiprocessing.set_start_method('spawn', force = True)
    
    experiment_data = {'runs_error': [], 'arrival_times': [], 'trajectories': []}
    
    print(f"Avvio di {NUM_RUNS} simulazioni Monte Carlo in PARALLELO...")
    start_time = time()

    cores_to_use = max(1, os.cpu_count() - 1)
    print(f"Utilizzando {cores_to_use} core su {os.cpu_count()}..\n")

    # Creiamo un "Pool" di processi. Usando max_workers=None Python usa tutti i core disponibili.
    with concurrent.futures.ProcessPoolExecutor(max_workers = cores_to_use) as executor:
        
        # Sottomettiamo tutti i task in blocco
        futures = [executor.submit(run_single_simulation, i) for i in range(NUM_RUNS)]
        
        runs_completed = 0
        
        # as_completed restituisce un task non appena finisce (non in ordine)
        for future in concurrent.futures.as_completed(futures):
            # Otteniamo i dati restituiti dal return della funzione
            error_hist, arrivals, trajectories = future.result()
            
            experiment_data['runs_error'].append(error_hist)
            experiment_data['arrival_times'].append(arrivals)
            experiment_data['trajectories'].append(trajectories)
            
            # --- BARRA DI PROGRESSO GLOBALE SULLA STESSA RIGA ---
            runs_completed += 1
            percent = int((runs_completed / NUM_RUNS) * 100)
            blocchi_pieni = int(percent / 5)
            barra = '█' * blocchi_pieni + '-' * (20 - blocchi_pieni)
            
            sys.stdout.write(f"\rCompletamento: |{barra}| {percent}% ({runs_completed}/{NUM_RUNS} runs completate)")
            sys.stdout.flush()

    # Tempo totale
    elapsed = time() - start_time
    
    with open('experiment_results_final1.pkl', 'wb') as f:
        pickle.dump(experiment_data, f)
        
    print(f"\nTutte le simulazioni sono concluse in {elapsed:.1f} secondi!")
    print("Dati salvati con successo in 'experiment_results_final1.pkl'")


