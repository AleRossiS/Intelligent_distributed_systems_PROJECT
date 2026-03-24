import matplotlib.pyplot as plt
import numpy as np
from viz_utility.robot_class import Robot
from viz_utility.environment_class import Environment
from voronoi_cells import VoronoiCell
from motion_functions import MotionController
from broadcast_and_consensus import broadcast_data

import config as cfg

plt.close('all')

# Inizializzazione
env = Environment()
env.add_walls()
env.add_obstacle([(2, 0), (4, 0), (4, 2), (2, 2)])
env.add_obstacle([(-6, -3), (-2, -3), (-2, -2.5), (-6, -2.5)])
env.add_obstacle([
    # --- BRACCIO SUPERIORE ESTERNO ---
    (0.8, -1.8), (1.2, -2.0), (1.6, -2.3), (2.0, -2.6),
    # --- DORSO ESTERNO (Guarda in basso a destra) ---
    (2.0, -2.9), (1.8, -3.3), (1.5, -3.9), (1.2, -4.4),
    # --- BRACCIO INFERIORE ESTERNO ---
    (0.8, -4.4), (0.2, -4.1), (-0.3, -3.8),
    # --- BRACCIO INFERIORE INTERNO (Rientro) ---
    (0.0, -3.6), (0.3, -3.6), (0.6, -3.8),
    # --- LA "PANCIA" DELLA C (Curva interna) ---
    (1.0, -3.8), (1.3, -3.5), (1.4, -3.1), 
    # --- BRACCIO SUPERIORE INTERNO (Ritorno alla punta) ---
    (1.2, -2.7), (0.9, -2.3), (0.7, -2.1)
])

# --- IL PICCOLO CERCHIO ---
cx, cy = -5.0, 7.0  # Coordinate del centro
raggio = 1.0       # Dimensione del raggio
num_lati = 20      # Risoluzione del cerchio

cerchio = [
    (cx + raggio * np.cos(2 * np.pi * i / num_lati), 
     cy + raggio * np.sin(2 * np.pi * i / num_lati)) 
    for i in range(num_lati)
]
env.add_obstacle(cerchio)

treasure_pos = (3, 3) # Posizione del tesoro
env.add_target_object(cx=treasure_pos[0], cy=treasure_pos[1])

obstacles_list = env.get_obstacle_polygons()

# SETUP PER IL TEST: 
r1 = Robot(robot_id=1, x=-1.0, y=3.0, color='tab:red')
r2 = Robot(robot_id=2, x=0.0, y=1.0, color='tab:blue')

env.add_robot(r1)
env.add_robot(r2)

# Inizializziamo il gestore di Voronoi
vor_r1 = VoronoiCell()
vor_r2 = VoronoiCell()

ctrl1 = MotionController()
ctrl2 = MotionController()

dt = 0.1 # Step temporale

# Simulazione
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))

for i in range(20000):

    # 1. AGGIORNA LA MEMORIA DEI ROBOT CON LE POSIZIONI ALTRUI
    for current_robot in env.robots:
        other_positions_dict = {}
        for other_robot in env.robots:
            if current_robot.robot_id != other_robot.robot_id:
                other_positions_dict[other_robot.robot_id] = (other_robot.state[0], other_robot.state[1])
        
        current_robot.other_robots_positions = other_positions_dict

    # 2. CALCOLIAMO I NUOVI VORONOI E I CENTROIDI
    pos = np.array([r.state[:2] for r in env.robots]) 
    
    cella1, centroid1 = vor_r1.compute_and_clip(pos, robot_index=0, obstacles=obstacles_list) 
    cella2, centroid2 = vor_r2.compute_and_clip(pos, robot_index=1, obstacles=obstacles_list) 
    
    # 3. CALCOLIAMO I COMANDI DI CONTROLLO
    u1 = ctrl1.get_control(r1, dt=0.1, voronoi=cella1[0], obstacles_list=obstacles_list, treasure_pos=treasure_pos) 
    u2 = ctrl2.get_control(r2, dt=0.1, voronoi=cella2[0], obstacles_list=obstacles_list, treasure_pos=treasure_pos) 



    # LOGICA DI COMUNICAZIONE ASINCRONA (MAPPA E POSIZIONI)

    for r in env.robots:
        # Il robot incrementa il suo clock e decide se è ora di trasmettere
        if r.tick_timer():
            broadcast_data(r, env.robots)
            r.is_transmitting = True
            print(f"[RETE] Clock raggiunto: Robot {r.robot_id} sta trasmettendo.")
        else:
            r.is_transmitting = False



    # LOGICA DI COMUNICAZIONE RADIO (RICHIESTA AIUTO)
    for current_robot in env.robots:
        id_chiamato = getattr(current_robot, 'id_to_call', None)
        
        if id_chiamato is not None:
            for compagno in env.robots:
                if compagno.robot_id == id_chiamato:
                    compagno.state_mode = "JOINING"
                    compagno.rendezvous_pos = current_robot.rendezvous_pos 
                    compagno.target_treasure_pos = current_robot.target_treasure_pos
                    print(f"[COMUNICAZIONE] Robot {current_robot.robot_id} ha chiamato Robot {compagno.robot_id} in aiuto sul punto originale!")
            
            current_robot.id_to_call = None
    # -----------------------------------------------------------------

    # 4. AGGIORNO LA POSIZIONE DEI ROBOT
    r1.update_pose(u1, dt)
    r2.update_pose(u2, dt)
    
    # 5. DISEGNIAMO AMBIENTE E ROBOT
    env.render(ax)
    
    # 6. DISEGNIAMO LE NUOVE CELLE DI VORONOI E I CENTROIDI
    vor_r1.plot_cells(ax, polygons=cella1, centroids=centroid1, cell_color='#ffffcc')
    vor_r2.plot_cells(ax, polygons=cella2, centroids=centroid2, cell_color='#ffffcc')

    # 7. MANTENIAMO FISSI I LIMITI DELLA TELECAMERA
    ax.set_xlim(cfg.X_LIM) 
    ax.set_ylim(cfg.Y_LIM)

    # Stampa di debug
    if i % 20 == 0:
        print(f"Step {i} | R1: {r1.state_mode} | R2: {r2.state_mode}")

    # 8. AGGIORNIAMO IL GRAFICO
    plt.pause(0.005) 

plt.ioff()