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

SWARM_SIZE = 1

# =================================================================
# FUNZIONE DELLA SINGOLA SIMULAZIONE (Eseguita dai vari Core)
# =================================================================
def run_single_simulation(run_id):
    """Questa funzione esegue una simulazione completa in isolamento e restituisce i dati."""

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

    robot_configs = [
        {'x': -7.0, 'y': 3.0, 'color': 'tab:red'},
        {'x': -6.0, 'y': 1.0, 'color': 'tab:blue'},
        {'x': -8.5, 'y': 2.5, 'color': 'tab:green'},
        {'x': -8.0, 'y': 5.0, 'color': 'tab:orange'},  # Posizione Robot 4
        {'x': -10.0, 'y': 3.0, 'color': 'tab:purple'}  # Posizione Robot 5
    ]

    robots = []
    voronois = []
    controllers = []
    
    for i in range(SWARM_SIZE):
        conf = robot_configs[i]
        r = Robot(robot_id=i+1, x=conf['x'], y=conf['y'], color=conf['color'])
        env.add_robot(r)
        robots.append(r)
        voronois.append(VoronoiCell())
        controllers.append(MotionController())

    dt = 0.1

    for current_robot in env.robots:
        for other_robot in env.robots:
            if current_robot.robot_id != other_robot.robot_id:
                current_robot.other_robots_positions[other_robot.robot_id] = (other_robot.state[0], other_robot.state[1])

    step_counter = 0
    run_error_history = []

    run_arrival_times = [None] * SWARM_SIZE 
    run_first_finder = None

    while step_counter < MAX_STEPS:

        real_obs_map = env.get_obstacle_polygons()
        inflated_obs_map = env.get_inflated_obstacles()

        controls = []

        for i, r in enumerate(robots):
            # Gestione dinamica: se ci sono altri robot usiamo vstack, altrimenti passiamo solo noi stessi
            other_pos = list(r.other_robots_positions.values())
            if len(other_pos) > 0:
                pos_r = np.vstack((r.state[:2], other_pos))
            else:
                pos_r = np.array([r.state[:2]]) # Passa un array 2D con un solo elemento
            cella, _ = voronois[i].compute_and_clip(r.true_state, r.state, pos_r, robot_index=0, obstacles=inflated_obs_map)
            
            if not hasattr(r, 'simulation_time'): r.simulation_time = 0.0
            r.simulation_time += dt
            
            u = controllers[i].get_control(r, dt, cella[0], inflated_obs_map, env.target_object)
            controls.append(u)

        for r in env.robots:
            if r.tick_timer(): broadcast_data(r, env.robots)

        for i, r in enumerate(robots):
            z_camera = global_camera.get_measurement(r.true_state, real_obs_map)
            r.update_pose(controls[i], dt, camera_measurement=z_camera)

        # RACCOLTA DATI
        sensed_count = sum([1 for r in robots if getattr(r, 'has_sensed_treasure', False)])

        if run_first_finder is None:
            # Controlla chi ha il flag True
            finders = [r.robot_id for r in robots if getattr(r, 'has_sensed_treasure', False)]
            if len(finders) > 0:
                run_first_finder = finders[0]
            
        # Registra tempi di arrivo
        for n in range(SWARM_SIZE):
            if sensed_count >= (n + 1) and run_arrival_times[n] is None:
                run_arrival_times[n] = step_counter


        estimates = [getattr(r, 'consensus_treasure_pos', None) for r in robots]
        valid_estimates = [e for e in estimates if e is not None]
        
        if len(valid_estimates) > 0:
            swarm_avg = np.mean(valid_estimates, axis=0)
            error = np.linalg.norm(swarm_avg - true_treasure_center)
            run_error_history.append(error)
        else:
            run_error_history.append(np.nan)
            
        step_counter += 1

    arrival_times = [t if t is not None else MAX_STEPS for t in run_arrival_times]

    # Estraiamo le traiettorie dell'intera run
    run_trajectories = []
    for r in robots:
        run_trajectories.append({
            'id': r.robot_id,
            'color': r.color,
            'x': r.history_true_x,
            'y': r.history_true_y
        })
    
    # Ritorna i dati al processo principale
    return run_error_history, arrival_times, run_trajectories, run_first_finder


# =================================================================
# GESTORE MULTIPROCESSING E BARRA DI CARICAMENTO (Main Process)
# =================================================================
if __name__ == '__main__':

    multiprocessing.set_start_method('spawn', force = True)
    
    experiment_data = {'runs_error': [], 'arrival_times': [], 'trajectories': [], 'first_finders': []}
    
    print(f"Avvio di {NUM_RUNS} simulazioni in PARALLELO con {SWARM_SIZE} robot...")
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
            error_hist, arrivals, trajectories, first_finder = future.result()
            
            experiment_data['runs_error'].append(error_hist)
            experiment_data['arrival_times'].append(arrivals)
            experiment_data['trajectories'].append(trajectories)
            experiment_data['first_finders'].append(first_finder)
            
            # barra di progressione
            runs_completed += 1
            percent = int((runs_completed / NUM_RUNS) * 100)
            blocchi_pieni = int(percent / 5)
            barra = '█' * blocchi_pieni + '-' * (20 - blocchi_pieni)
            
            sys.stdout.write(f"\rCompletamento: |{barra}| {percent}% ({runs_completed}/{NUM_RUNS} runs completate)")
            sys.stdout.flush()

    # Tempo totale
    elapsed = time() - start_time
    
    filename = f'experiments_pkl_files/experiment_results_{SWARM_SIZE}_robots.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(experiment_data, f)
        
    print(f"\nTutte le simulazioni sono concluse in {elapsed:.1f} secondi!")
    print(f"Dati salvati con successo in '{filename}'")


