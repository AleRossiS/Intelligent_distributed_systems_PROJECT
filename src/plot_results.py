import os
import pickle 
import numpy as np
import matplotlib.pyplot as plt

# 1. Carica i Dati
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
FILENAME = os.path.join(RESULTS_DIR, 'data', 'experiment_results_1_robots.pkl')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

try:
    with open(FILENAME, 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"Errore: Il file {FILENAME} non è stato trovato. Assicurati di aver lanciato la simulazione.")
    exit()

runs_error = data['runs_error']
arrival_times = data['arrival_times']

# DEDUCIAMO LA DIMENSIONE DELLO SCIAME DAI DATI STESSI
SWARM_SIZE = len(arrival_times[0])
print(f"Analisi dati avviata per uno sciame di {SWARM_SIZE} robot...")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

# Palette di colori espansa per supportare fino a 5 robot
ROBOT_COLORS = ['lightcoral', 'lightblue', 'lightgreen', 'moccasin', 'thistle']
ROBOT_LABELS = [f'Robot {i+1}' for i in range(SWARM_SIZE)]

# =========================================================
# GRAFICO 1: Errore di Stima del Tesoro (Sensor Fusion)
# =========================================================
# L'array runs_error contiene np.nan (vuoto) prima della scoperta.
# Per mostrare la convergenza, allineiamo tutte le run partendo
# dal momento in cui ogni sciame ha trovato il tesoro (Time=0)
OBSERVATION_WINDOW = 500
aligned_errors = []

for run_err in runs_error:
    # Prendi solo la parte della curva DOPO che il tesoro è stato scoperto
    valid_errs = [e for e in run_err if not np.isnan(e)]
    if len(valid_errs) > 0:
        if len(valid_errs) < OBSERVATION_WINDOW:
            valid_errs += [valid_errs[-1]] * (OBSERVATION_WINDOW - len(valid_errs))
        aligned_errors.append(valid_errs[:OBSERVATION_WINDOW])

if aligned_errors:
    err_matrix = np.array(aligned_errors)
    # Calcolo Media e Deviazione Standard statistica
    mean_err = np.mean(err_matrix, axis=0)
    std_err = np.std(err_matrix, axis=0)
    
    # Converti gli step in secondi (dt = 0.1)
    time_axis = np.arange(OBSERVATION_WINDOW) * 0.1
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, mean_err, 'b-', linewidth=2.5, label='Mean error')
    plt.fill_between(time_axis, mean_err - std_err, mean_err + std_err, color='b', alpha=0.2, label=r'$\pm 1\sigma$')
    
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    plt.title(f"Consensus convergence ({SWARM_SIZE} Robots)")
    plt.xlabel("Time after first detection [s]")
    plt.ylabel("Target position estimation [m]")
    plt.ylim(-0.0, max(mean_err)+ 0.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/grafico_errore_{SWARM_SIZE}_ROBOTS.png', dpi=300)
    print(f"Grafico 1: 'grafico_errore_{SWARM_SIZE}_ROBOTS.png' salvato.")

# =========================================================
# GRAFICO 2: Time to First Detection (Boxplot)
# =========================================================

times_sec = []
box_labels = []

for i in range(SWARM_SIZE):
    # Estraiamo l'i-esimo elemento di ogni simulazione e lo convertiamo in secondi
    t_list = [t[i] * 0.1 for t in arrival_times if t[i] is not None]
    times_sec.append(t_list)
    # Creiamo etichette come "Robot 1 (FTT)", "Robot 2 (STT)" ecc...
    suffix = "First" if i == 0 else "Second" if i == 1 else "Third" if i == 2 else "Fourth" if i == 3 else "Fifth"
    box_labels.append(f'{suffix}')

plt.figure(figsize=(8, 6))

box = plt.boxplot(times_sec, tick_labels=box_labels, patch_artist=True)

for patch, color in zip(box['boxes'], ROBOT_COLORS[:SWARM_SIZE]):
    patch.set_facecolor(color)
    
plt.title(f"Time distribution for arrival at the Treasure ({SWARM_SIZE} Robots)")
plt.ylabel("Time [s]")
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/grafico_arrivi_{SWARM_SIZE}_ROBOTS.png', dpi=300)
print(f"Grafico 2: 'grafico_arrivi_{SWARM_SIZE}_ROBOTS.png' salvato.")

# =========================================================
# GRAFICO 3: Traiettorie Sovrapposte (Heat Map)
# =========================================================
# Recuperiamo le traiettorie dal file pkl
trajectories_data = data.get('trajectories', [])

if trajectories_data:
    plt.figure(figsize=(10, 8))
    
    # Disegniamo i muri/ostacoli per dare contesto (approssimati)
    # È utile mettere le stesse coordinate che hai nell'ambiente
    obstacles = [
        [(2, 0), (4, 0), (4, 2), (2, 2)],
        [(-9, -2), (-5, -2), (-5, -1.5), (-9, -1.5)],
        [(-5, -5.5), (-1, -5.5), (-1, -5), (-5, -5)]
    ]
    for obs in obstacles:
        poly = plt.Polygon(obs, color='gray', alpha=0.5)
        plt.gca().add_patch(poly)
    
    cerchio1 = plt.Circle((-5.0, 6.0), 1.0, color='gray', alpha=0.5)
    plt.gca().add_patch(cerchio1)
    
    cerchio2 = plt.Circle((7.0, -6.0), 1.0, color='gray', alpha=0.5)
    plt.gca().add_patch(cerchio2)
    
    # E il tesoro
    circle = plt.Circle((9, 4), 0.5, color='gold', alpha=0.3, label="Treasure Area")
    plt.gca().add_patch(circle)

    # Iteriamo su TUTTE le 100 simulazioni
    for run_idx, run_traj in enumerate(trajectories_data):
        for robot_data in run_traj:
            color = robot_data['color']
            lbl = f"R{robot_data['id']}" if run_idx == 0 else ""
            plt.plot(robot_data['x'], robot_data['y'], color=color, alpha=0.5, linewidth=0.5, label=lbl)

    plt.title(f"Swarm Exploration Coverage ({SWARM_SIZE} Robots, 100 Runs)")
    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    
    # Rimuovi i duplicati dalla legenda causati dal loop
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = plt.legend(by_label.values(), by_label.keys(), loc='upper left')

    for line in leg.get_lines():
        line.set_alpha(1.0)  # Rendi le linee della legenda completamente opache
        line.set_linewidth(2.0)  # Rendi le linee della legenda più spesse
    
    plt.xlim(-15, 15)
    plt.ylim(-10, 10)
    plt.axis('equal') # Mantiene le proporzioni quadrate
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/grafico_traiettorie_{SWARM_SIZE}_ROBOTS.png', dpi=300)
    print(f"Grafico 3: 'grafico_traiettorie_{SWARM_SIZE}_ROBOTS.png' salvato.")

    # =========================================================
# GRAFICO 4: Distribuzione delle Scoperte (Pie Chart)
# =========================================================
first_finders = data.get('first_finders', [])

if first_finders:
    wins = [first_finders.count(i+1) for i in range(SWARM_SIZE)]

    valid_wins = []
    valid_labels = []
    valid_colors = []

    for i in range(SWARM_SIZE):
        if wins[i] > 0:
            valid_wins.append(wins[i])
            valid_labels.append(ROBOT_LABELS[i])
            valid_colors.append(ROBOT_COLORS[i])

    if valid_wins:
        plt.figure(figsize=(7, 7))
        explode = tuple([0.05] * len(valid_wins))
        
        plt.pie(valid_wins, labels=valid_labels, colors=valid_colors, autopct='%1.1f%%', 
                startangle=140, explode=explode, shadow=True)
        
        plt.title(f"First Detection Distribution ({SWARM_SIZE} Robots)")
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/grafico_scoperte_{SWARM_SIZE}_ROBOTS.png', dpi=300)
        print(f"Grafico 4: 'grafico_scoperte_{SWARM_SIZE}_ROBOTS.png' salvato.")

plt.show()