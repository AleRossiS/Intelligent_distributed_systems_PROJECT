# import matplotlib.pyplot as plt
import numpy as np
import pygame
import sys
from viz_utility.robot_class import Robot
from viz_utility.environment_class import Environment
from voronoi_cells import VoronoiCell
from motion_and_mapping import MotionController
from broadcast_and_consensus import broadcast_data

from ekf_localization import GlobalCamera

import config as cfg

pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulazione Robotica con Voronoi e EKF")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 12, bold=True)

# Mappatura dei colori da Matplotlib a RGB per Pygame
COLORS = {
    'tab:red': (214, 39, 40),
    'tab:blue': (31, 119, 180),
    'tab:green': (44, 160, 44),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gold': (255, 215, 0),
    'magenta': (255, 0, 255),
    'gray': (150, 150, 150),
    'light_gray': (245, 245, 245)
}

def to_px(x, y):
    """Converte le coordinate matematiche (es. -10 a +10) in pixel dello schermo. (0,0) è in alto a sinistra. in pygame l'asse Y è invertito rispetto alla rappresentazione matematica, quindi bisogna fare un'inversione. """
    px = int((x - cfg.X_LIM[0]) / (cfg.X_LIM[1] - cfg.X_LIM[0]) * WIDTH)
    py = int((cfg.Y_LIM[1] - y) / (cfg.Y_LIM[1] - cfg.Y_LIM[0]) * HEIGHT) # Asse Y invertito in Pygame!
    return px, py


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

# SETUP TELECAMERA GLOBALE (Piazzata in alto a destra)
global_camera = GlobalCamera(x=9.0, y=9.0, noise_std_x=0.05, noise_std_y=0.05, noise_std_theta=0.02)



# SETUP PER IL TEST: 
r1 = Robot(robot_id=1, x=-1.0, y=3.0, color='tab:red')
r2 = Robot(robot_id=2, x=0.0, y=1.0, color='tab:blue')
r3 = Robot(robot_id=3, x=-7.5, y=-7.5, color='tab:green') 

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


#Main loop at 60 FPS
step_counter = 0
running = True

# Dizionario per tenere traccia delle animazioni delle onde radio
# Formato: {robot_id: progresso_animazione_da_0_a_1}
wave_animations = {}

while running:
    # Gestione eventi (es. chiusura finestra)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # # Opzionale: Riavvia con 'R'
        # if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
        #     print("Resetting simulation...")
        #     # (Codice di reset non implementato qui per brevità)
    
    # Logica di simulazione
    pos_r1 = np.vstack((r1.state[:2], list(r1.other_robots_positions.values())))
    pos_r2 = np.vstack((r2.state[:2], list(r2.other_robots_positions.values())))
    pos_r3 = np.vstack((r3.state[:2], list(r3.other_robots_positions.values())))

    real_obs_map = env.get_obstacle_polygons()
    inflated_obs_map = env.get_inflated_obstacles()

    cella1, centroid1 = vor_r1.compute_and_clip(pos_r1, robot_index=0, obstacles=inflated_obs_map)
    cella2, centroid2 = vor_r2.compute_and_clip(pos_r2, robot_index=0, obstacles=inflated_obs_map)
    cella3, centroid3 = vor_r3.compute_and_clip(pos_r3, robot_index=0, obstacles=inflated_obs_map)

    u1 = ctrl1.get_control(r1, env.robots, dt, cella1[0], real_obs_map, inflated_obs_map, env.target_object)
    u2 = ctrl2.get_control(r2, env.robots, dt, cella2[0], real_obs_map, inflated_obs_map, env.target_object)
    u3 = ctrl3.get_control(r3, env.robots, dt, cella3[0], real_obs_map, inflated_obs_map, env.target_object)

    # Rete e trasmissione
    for r in env.robots:
        # Inizia una nuova animazione per questo robot!
        if r.robot_id not in wave_animations:
            wave_animations[r.robot_id] = []

        if r.tick_timer():
            broadcast_data(r, env.robots)
            r.is_transmitting = True

            # Generiamo seconda onda se la precedente è già a metà del suo ciclo (progress > 0.4) per un effetto di trasmissione più continuo
            waves = wave_animations[r.robot_id]
            if len(waves) == 0 or waves[-1] > 0.4:
                waves.append(0.0) # Inizia una nuova onda

            
        else:
            r.is_transmitting = False

    # fisica e kalman
    for r, u in zip([r1, r2, r3], [u1, u2, u3]):
        z_camera = global_camera.get_measurement(r.true_state, real_obs_map)
        r.update_pose(u, dt, camera_measurement=z_camera)

    # Rendering
    screen.fill(COLORS['light_gray']) # Colore di sfondo

    #Disegna voronoi
    vor_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for cella in [cella1, cella2, cella3]:
        for poly in cella:
            if poly and not poly.is_empty:
                polygons_to_draw = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
                for poly in polygons_to_draw:
                    pts = [to_px(x, y) for x, y in poly.exterior.coords]
                    pygame.draw.polygon(vor_surface, (255, 255, 204, 70), pts) #giallo trasparente
                    pygame.draw.polygon(vor_surface, (200, 200, 200), pts, 2) #bordo grigio
    screen.blit(vor_surface, (0, 0))

    # Disegna ostacoli
    for obs in env.obstacles:
        pts = [to_px(x, y) for x, y in obs]
        pygame.draw.polygon(screen, COLORS['gray'], pts)
        pygame.draw.polygon(screen, COLORS['black'], pts, 2)

    # Disegna il tesoro
    if env.target_object:
        pts = [to_px(x, y) for x, y in env.target_object.exterior.coords]
        pygame.draw.polygon(screen, COLORS['gold'], pts)
        pygame.draw.polygon(screen, COLORS['black'], pts, 2)

    # Disegna telecamera globale
    cam_px = to_px(global_camera.pos.x, global_camera.pos.y)
    pygame.draw.rect(screen, COLORS['magenta'], (cam_px[0]-5, cam_px[1]-5, 10, 10))

    # Disegna Robot e loro Annotazioni
    for r, z_camera in zip(env.robots, [z_camera for z_camera in [global_camera.get_measurement(r_test.true_state, real_obs_map) for r_test in env.robots]]):
        color = COLORS.get(r.color, COLORS['black'])
        tx, ty = to_px(r.true_state[0], r.true_state[1])
        radius_px = int(r.radius / (cfg.X_LIM[1] - cfg.X_LIM[0]) * WIDTH)

        # === 0. RAGGIO DELLA TELECAMERA (Line of Sight) ===
        if z_camera is not None:
            # Il robot è visibile! Disegna un raggio di tracciamento azzurro
            pygame.draw.line(screen, (0, 200, 255), cam_px, (tx, ty), 1)
            camera_status = "TRACKING"
            status_color = (0, 150, 0) # Verde scuro
        else:
            # Il robot è nascosto dietro un ostacolo!
            camera_status = "NO SIGNAL"
            status_color = (255, 0, 0) # Rosso allarme

        # === 1. DISEGNA LE "X" SULLE FRONTIERE ISPEZIONATE ===
        cross_size = 5 
        for ix, iy in r.inspected_points:
            ipx, ipy = to_px(ix, iy)
            pygame.draw.line(screen, color, (ipx - cross_size, ipy - cross_size), (ipx + cross_size, ipy + cross_size), 2)
            pygame.draw.line(screen, color, (ipx + cross_size, ipy - cross_size), (ipx - cross_size, ipy + cross_size), 2)

        # === 2. Traccia Storica (Ultime 150 posizioni) ===
        if len(r.history_true_x) > 2:
            hist_pts = [to_px(hx, hy) for hx, hy in zip(r.history_true_x[-150:], r.history_true_y[-150:])]
            pygame.draw.lines(screen, color, False, hist_pts, 2)

        # === 3. Robot Stimato (EKF - Cerchio vuoto tratteggiato) ===
        ex, ey = to_px(r.state[0], r.state[1])
        pygame.draw.circle(screen, color, (ex, ey), radius_px, 1)
        
        # === 4. Robot Reale (Solido colorato) ===
        pygame.draw.circle(screen, color, (tx, ty), radius_px)
        pygame.draw.circle(screen, COLORS['black'], (tx, ty), radius_px, 2) 
        
        # Linea di direzione (Robot Reale)
        front_x = tx + radius_px * np.cos(r.true_state[2])
        front_y = ty - radius_px * np.sin(r.true_state[2]) 
        pygame.draw.line(screen, COLORS['black'], (tx, ty), (front_x, front_y), 3)

        # === 5. ANIMAZIONE ONDE RADIO ESPANSE ===
        if r.robot_id in wave_animations:
            active_waves = [] # Lista temporanea per le onde che non sono ancora svanite
            
            for progress in wave_animations[r.robot_id]:
                # L'onda si espande fino a 3 volte il raggio
                wave_radius = int(radius_px + (radius_px * 3 * progress))
                # Lo spessore sfuma gradualmente
                wave_thickness = max(1, int(3 - (progress * 3)))
                
                pygame.draw.circle(screen, COLORS['magenta'], (tx, ty), wave_radius, wave_thickness)
                
                # Aggiorna l'onda. Se è minore di 1.0 (100%), la teniamo per il prossimo frame
                new_progress = progress + 0.015
                if new_progress < 1.0:
                    active_waves.append(new_progress)
            
            # Salviamo solo le onde ancora in viaggio
            wave_animations[r.robot_id] = active_waves

        # === 6. TESTO E STATO ===
        # Info Stato Robot (es. SEARCHING)
        state_text = f"R{r.robot_id} {r.state_mode}"
        text_surf = font.render(state_text, True, COLORS['black'])
        text_rect = text_surf.get_rect(center=(tx, ty - radius_px - 25))
        pygame.draw.rect(screen, COLORS['white'], text_rect.inflate(4, 2))
        screen.blit(text_surf, text_rect)
        
        # Info Telecamera (TRACKING / NO SIGNAL)
        cam_surf = font.render(camera_status, True, status_color)
        cam_rect = cam_surf.get_rect(center=(tx, ty - radius_px - 10))
        pygame.draw.rect(screen, COLORS['white'], cam_rect.inflate(4, 2))
        screen.blit(cam_surf, cam_rect)

    # Aggiorna lo schermo e mantieni 60 FPS fisse
    pygame.display.flip()
    clock.tick(60)

    if step_counter % 20 == 0:
        print(f"Step {step_counter} | R1: {r1.state_mode} | R2: {r2.state_mode} | R3: {r3.state_mode}")
    step_counter += 1

pygame.quit()
sys.exit()



