import matplotlib.pyplot as plt
import numpy as np
from viz_utility.robot_class import Robot
from viz_utility.environment_class import Environment
import config as cfg
from motion_functions import LevyController


plt.close('all')

# 1. Inizializzazione Ambiente
env = Environment(xlim=cfg.X_LIM, ylim=cfg.Y_LIM)

# (Opzionale) Aggiungi gli ostacoli se vuoi vederli nello sfondo
obstacles_list = env.get_obstacle_polygons()

# 2. Creazione Robot
# ATTENZIONE: Assicurati che il tuo file viz_utility/robot_class.py 
# contenga il metodo get_levy_control() che abbiamo scritto prima!
r1 = Robot(1, x=-3, y=3, color='red')
r2 = Robot(2, x=3, y=-3, color='blue')

env.add_robot(r1)
env.add_robot(r2)


# 3. Setup Animazione
plt.ion() # Attiva la modalità interattiva
fig, ax = plt.subplots(figsize=(8,8))

dt = 0.1 # Step temporale
sim_steps = 1500 # Numero di frame dell'animazione

ctrl1 = LevyController()
ctrl2 = LevyController()

for i in range(sim_steps):
    # --- A. AGGIORNAMENTO FISICA E LOGICA ---
    # Calcolo il comando di velocità usando il Lévy Flight
    u1 = ctrl1.get_control(r1.state, dt=0.1)
    u2 = ctrl2.get_control(r2.state, dt=0.1)
    
    # Aggiorno la posizione del robot
    r1.update_pose(u1, dt)
    r2.update_pose(u2, dt)
        
    # --- B. DISEGNO ---
    # Il metodo render(ax) svuota l'asse e ridisegna ostacoli e robot
    env.render(ax)
    
    
    # --- D. AGGIORNAMENTO SCHERMO ---
    plt.pause(0.01) # Pausa breve per dare il tempo a Matplotlib di renderizzare

# Fine della simulazione, teniamo la finestra aperta
plt.ioff()
print("Simulazione Terminata!")
plt.show()