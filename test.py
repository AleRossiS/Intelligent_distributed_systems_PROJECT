import matplotlib.pyplot as plt
import numpy as np
from viz_utility.robot_class import Robot
from viz_utility.environment_class import Environment
from voronoi_cells import VoronoiCell
from motion_and_mapping import MotionController
from broadcast_and_consensus import broadcast_data

import config as cfg

plt.close('all')

# Inizializzazione
env = Environment()
env.add_walls()
#env.add_obstacle([(2, 0), (4, 0), (4, 2), (2, 2)])
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
r3 = Robot(robot_id=3, x=-2.0, y=4.0, color='tab:green') 

env.add_robot(r1)
env.add_robot(r2)
env.add_robot(r3)

# Inizializziamo il gestore di Voronoi
vor_r1 = VoronoiCell()
vor_r2 = VoronoiCell()
vor_r3 = VoronoiCell()

ctrl1 = MotionController()
ctrl2 = MotionController()
ctrl3 = MotionController()

dt = 0.1 # Step temporale

# INIZIALIZZAZIONE MEMORIA (Per non far crashare Voronoi al frame 0) 
for current_robot in env.robots:
    for other_robot in env.robots:
        if current_robot.robot_id != other_robot.robot_id:
            current_robot.other_robots_positions[other_robot.robot_id] = (other_robot.state[0], other_robot.state[1])

# Simulazione
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))

for i in range(20000):

    # 2. CALCOLIAMO I NUOVI VORONOI IN MODO DECENTRALIZZATO
    # Costruiamo la percezione del mondo per R1
    pos_r1 = [r1.state[:2]] # R1 mette se stesso al primo posto (indice 0)
    for other_pos in r1.other_robots_positions.values():
        pos_r1.append(other_pos)
    pos_r1 = np.array(pos_r1)
    
    # Costruiamo la percezione del mondo per R2
    pos_r2 = [r2.state[:2]] # R2 mette se stesso al primo posto (indice 0)
    for other_pos in r2.other_robots_positions.values():
        pos_r2.append(other_pos)
    pos_r2 = np.array(pos_r2)
    
        # Costruiamo la percezione del mondo per R2
    pos_r3 = [r3.state[:2]] # R3 mette se stesso al primo posto (indice 0)
    for other_pos in r3.other_robots_positions.values():
        pos_r3.append(other_pos)
    pos_r3 = np.array(pos_r3)
    
    # Calcolo Voronoi (notare che robot_index è diventato 0 per entrambi!)
    cella1, centroid1 = vor_r1.compute_and_clip(pos_r1, robot_index=0, obstacles=obstacles_list) 
    cella2, centroid2 = vor_r2.compute_and_clip(pos_r2, robot_index=0, obstacles=obstacles_list) 
    cella3, centroid3 = vor_r3.compute_and_clip(pos_r3, robot_index=0, obstacles=obstacles_list) 
    
    # 3. CALCOLIAMO I COMANDI DI CONTROLLO
    u1 = ctrl1.get_control(r1, env.robots, dt=0.1, voronoi=cella1[0], obstacles_list=obstacles_list, target_obj=env.target_object) 
    u2 = ctrl2.get_control(r2, env.robots, dt=0.1, voronoi=cella2[0], obstacles_list=obstacles_list, target_obj=env.target_object) 
    u3 = ctrl3.get_control(r3, env.robots, dt=0.1, voronoi=cella3[0], obstacles_list=obstacles_list, target_obj=env.target_object) 


    # LOGICA DI COMUNICAZIONE ASINCRONA (MAPPA E POSIZIONI)
    for r in env.robots:
        # Il robot incrementa il suo clock e decide se è ora di trasmettere
        if r.tick_timer():
            broadcast_data(r, env.robots)
            r.is_transmitting = True  # O 15 se hai usato l'animazione ad onde!
            print(f"[RETE] Clock raggiunto: Robot {r.robot_id} sta trasmettendo.")
        else:
            r.is_transmitting = False # (O la logica di decremento se hai le onde)
        

    # 4. AGGIORNO LA POSIZIONE DEI ROBOT
    r1.update_pose(u1, dt)
    r2.update_pose(u2, dt)
    r3.update_pose(u3, dt)
    # 5. DISEGNIAMO AMBIENTE E ROBOT
    env.render(ax)
    
    # 6. DISEGNIAMO LE NUOVE CELLE DI VORONOI E I CENTROIDI
    vor_r1.plot_cells(ax, polygons=cella1, centroids=centroid1, cell_color='#ffffcc')
    vor_r2.plot_cells(ax, polygons=cella2, centroids=centroid2, cell_color='#ffffcc')
    vor_r3.plot_cells(ax, polygons=cella3, centroids=centroid3, cell_color='#ffffcc')

    # 7. MANTENIAMO FISSI I LIMITI DELLA TELECAMERA
    ax.set_xlim(cfg.X_LIM) 
    ax.set_ylim(cfg.Y_LIM)

    # Stampa di debug
    if i % 20 == 0:
        print(f"Step {i} | R1: {r1.state_mode} | R2: {r2.state_mode} | R3: {r3.state_mode}")

    # 8. AGGIORNIAMO IL GRAFICO
    plt.pause(0.005) 

plt.ioff()