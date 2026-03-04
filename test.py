import matplotlib.pyplot as plt
import numpy as np
from viz_utility.robot_class import Robot
from viz_utility.environment_class import Environment
from voronoi_cells import VoronoiCell

import config as cfg

# Inizializzazione
env = Environment(xlim=cfg.X_LIM, ylim=cfg.Y_LIM)
env.set_zones(cfg.PICKUP_ZONE, cfg.DELIVERY_ZONE) # (x left corner, y left corner, width, height)
env.add_obstacle([(2, 0), (4, 0), (4, 2), (2, 2)])
env.add_obstacle([(-6, -3), (-2, -3), (-2, -2.5), (-6, -2.5)])

obstacles_list = env.get_obstacle_polygons()

# Creiamo i robot e li aggiungiamo all'ambiente
r1 = Robot(1, x=-3, y=3, color='red')
r2 = Robot(2, x=-2.2, y=1.5, color='blue')
r3 = Robot(3, x=-1.4, y=-0.8, color='green')
r4 = Robot(4, x=-4, y=-1.5, color='orange')
env.add_robot(r1)
env.add_robot(r2)
env.add_robot(r3)
env.add_robot(r4)

# Inizializziamo il gestore di Voronoi
vor_r1 = VoronoiCell(x_lim=cfg.X_LIM, y_lim=cfg.Y_LIM)
vor_r2 = VoronoiCell(x_lim=cfg.X_LIM, y_lim=cfg.Y_LIM)
vor_r3 = VoronoiCell(x_lim=cfg.X_LIM, y_lim=cfg.Y_LIM)
vor_r4 = VoronoiCell(x_lim=cfg.X_LIM, y_lim=cfg.Y_LIM)

# Simulazione
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))

for i in range(2000):
    # 1. AGGIORNAMENTO POSIZIONI ROBOT
    for r in env.robots:
        
        r.update_pose([0.4, 0.5], dt=0.1) # (v, w) e dt
    
    # 2. CALCOLIAMO I NUOVI VORONOI E I CENTROIDI
    # Estraiamo le nuove posizioni aggiornate
    pos = np.array([r.state[:2] for r in env.robots]) 
    
    # Calcoliamo celle e centroidi tramite i metodi della classe
    celle1, centroidi1 = vor_r1.compute_and_clip(pos, robot_index=0, obstacles=obstacles_list) 
    celle2, centroidi2 = vor_r2.compute_and_clip(pos, robot_index=1, obstacles=obstacles_list) 
    celle3, centroidi3 = vor_r3.compute_and_clip(pos, robot_index=2, obstacles=obstacles_list)
    celle4, centroidi4 = vor_r4.compute_and_clip(pos, robot_index=3, obstacles=obstacles_list)
    # 3. DISEGNIAMO AMBIENTE E ROBOT (questo fa il "clear" dello schermo)
    env.render(ax)
    
    # 4. DISEGNIAMO LE NUOVE CELLE DI VORONOI E I CENTROIDI
    # Passiamo l'asse, le celle, i centroidi e il colore giallino!
    vor_r1.plot_cells(ax, polygons=celle1, centroids=centroidi1, cell_color='#ffffcc')
    vor_r2.plot_cells(ax, polygons=celle2, centroids=centroidi2, cell_color='#ffffcc')
    vor_r3.plot_cells(ax, polygons=celle3, centroids=centroidi3, cell_color='#ffffcc')
    vor_r4.plot_cells(ax, polygons=celle4, centroids=centroidi4, cell_color='#ffffcc')

    # 5. MANTENIAMO FISSI I LIMITI DELLA TELECAMERA
    ax.set_xlim(-10, 10)
    ax.set_ylim(-8, 8)

    # 6. AGGIORNIAMO IL GRAFICO
    plt.pause(0.0001) 

plt.ioff()
plt.show()