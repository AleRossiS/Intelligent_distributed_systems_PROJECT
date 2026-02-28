import matplotlib.pyplot as plt
import numpy as np
from viz_utility.robot_class import Robot
from viz_utility.environment_class import Environment
from Intelligent_distributed_systems_PROJECT.voronoi_cells import VoronoiCell

# Inizializzazione
env = Environment(xlim=(-10, 10), ylim=(-8, 8))
env.set_zones((-1.5, -8, 3, 2), (-1.5, 8-2, 3, 2)) # (x left corner, y left corner, width, height)
env.add_obstacle(2, 0, 2, 2)
env.add_obstacle(-6, -3, 4, 0.5)

# Creiamo i robot e li aggiungiamo all'ambiente
r1 = Robot(1, x=-3, y=3, color='red')
r2 = Robot(2, x=3, y=-3, color='blue')
r3 = Robot(3, x=-6, y=-2, color='green')
r4 = Robot(4, x=7, y=2, color='orange')
env.add_robot(r1)
env.add_robot(r2)
env.add_robot(r3)
env.add_robot(r4)

# Inizializziamo il gestore di Voronoi
vor = VoronoiCell(x_lim=(-10, 10), y_lim=(-8, 8))

# Simulazione
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))

for i in range(200):
    # 1. AGGIORNAMENTO POSIZIONI ROBOT
    for r in env.robots:
        
        r.update_pose([0.4, 0.5], dt=0.1)
    
    # 2. CALCOLIAMO I NUOVI VORONOI E I CENTROIDI
    # Estraiamo le nuove posizioni aggiornate
    pos = np.array([r.state[:2] for r in env.robots]) 
    
    # Calcoliamo celle e centroidi tramite i metodi della classe
    celle = vor.compute_and_clip(pos)
    centroidi = vor.get_centroids(celle)
    
    # 3. DISEGNIAMO AMBIENTE E ROBOT (questo fa il "clear" dello schermo)
    env.render(ax)
    
    # 4. DISEGNIAMO LE NUOVE CELLE DI VORONOI E I CENTROIDI
    # Passiamo l'asse, le celle, i centroidi e il colore giallino!
    vor.plot_cells(ax, polygons=celle, centroids=centroidi, cell_color='#ffffcc')

    # 5. MANTENIAMO FISSI I LIMITI DELLA TELECAMERA
    ax.set_xlim(-10, 10)
    ax.set_ylim(-8, 8)

    # 6. AGGIORNIAMO IL GRAFICO
    plt.pause(0.01) 

plt.ioff()
plt.show()