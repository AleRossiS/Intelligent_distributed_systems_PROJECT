import matplotlib.pyplot as plt
import numpy as np
from viz_utility.robot_class import Robot
from viz_utility.environment_class import Environment
from voronoi_cells import VoronoiCell
from motion_functions import MotionController

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

obstacles_list = env.get_obstacle_polygons()

# Creiamo i robot e li aggiungiamo all'ambiente
r1 = Robot(1, x=-3, y=3, color='tab:red')
r2 = Robot(2, x=-2.2, y=1.5, color='tab:blue')

env.add_robot(r1)
env.add_robot(r2)


# Inizializziamo il gestore di Voronoi
vor_r1 = VoronoiCell()
vor_r2 = VoronoiCell()



ctrl1 = MotionController()
ctrl2 = MotionController()

# (Opzionale) Visualizza la distribuzione di Lévy per verificare che sia corretta
#ctrl1.plot_levy_distribution()
dt = 0.1 # Step temporale

# Simulazione
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))

for i in range(2000):
    # 1. AGGIORNAMENTO POSIZIONI ROBOT
    # Calcolo il comando di velocità usando il Lévy Flight

    
    # 2. CALCOLIAMO I NUOVI VORONOI E I CENTROIDI
    # Estraiamo le nuove posizioni aggiornate
    pos = np.array([r.state[:2] for r in env.robots]) 
    
    # Calcoliamo celle e centroidi tramite i metodi della classe
    cella1, centroid1 = vor_r1.compute_and_clip(pos, robot_index=0, obstacles=obstacles_list) 
    cella2, centroid2 = vor_r2.compute_and_clip(pos, robot_index=1, obstacles=obstacles_list) 
    
    
    u1 = ctrl1.levy_control(r1.state, dt=0.1, voronoi=cella1[0]) 
    u2 = ctrl2.levy_control(r2.state, dt=0.1, voronoi=cella2[0]) 

    # Aggiorno la posizione del robot
    r1.update_pose(u1, dt)
    r2.update_pose(u2, dt)
    # 3. DISEGNIAMO AMBIENTE E ROBOT (questo fa il "clear" dello schermo)
    env.render(ax)
    
    # 4. DISEGNIAMO LE NUOVE CELLE DI VORONOI E I CENTROIDI
    # Passiamo l'asse, le celle, i centroidi e il colore giallino!
    vor_r1.plot_cells(ax, polygons=cella1, centroids=centroid1, cell_color='#ffffcc')
    vor_r2.plot_cells(ax, polygons=cella2, centroids=centroid2, cell_color='#ffffcc')

    # 5. MANTENIAMO FISSI I LIMITI DELLA TELECAMERA
    #margine_vista = 1.0
    ax.set_xlim(cfg.X_LIM[0])
    ax.set_ylim(cfg.Y_LIM[0])

    # 6. AGGIORNIAMO IL GRAFICO
    plt.pause(0.005) 

plt.ioff()
plt.show()