import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. Carica i Dati
with open('experiment_results_final2.pkl', 'rb') as f:
    data = pickle.load(f)

runs_error = data['runs_error']
arrival_times = data['arrival_times']

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

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
    
    plt.title("Consensus convergance")
    plt.xlabel("Time after first detection [s]")
    plt.ylabel("Target position estimation [m]")
    plt.ylim(-0.0, max(mean_err)+ 0.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig('grafico_errore.png', dpi=300)
    print("Grafico 1: 'grafico_errore.png' salvato.")

# =========================================================
# GRAFICO 2: Time to First Detection (Boxplot)
# =========================================================
# Convertiamo i TTFD in secondi
t1_sec = [t[0] * 0.1 for t in arrival_times]
t2_sec = [t[1] * 0.1 for t in arrival_times]
t3_sec = [t[2] * 0.1 for t in arrival_times]

plt.figure(figsize=(8, 6))

box = plt.boxplot([t1_sec, t2_sec, t3_sec], tick_labels=['Robot 1 (FTT)', 'Robot 2 (STT)', 'Robot 3 (TTT)'], patch_artist=True)

colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    
plt.title("Time distribution for arrival at the Treasure")
plt.ylabel("Time [s]")
plt.tight_layout()
plt.savefig('grafico_arrivi.png', dpi=300)
print("Grafico 2: 'grafico_arrivi.png' salvato.")

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
        # Ogni simulazione ha i percorsi di r1, r2, r3
        for robot_data in run_traj:
            color = robot_data['color']
            
            # Se vuoi che il rosso, blu e verde siano più scuri, lascia il loro colore.
            # Se vuoi un effetto più "heat map" uniforme, potresti forzare un colore unico (es. 'black')
            # Qui usiamo i loro colori ma li rendiamo super trasparenti (alpha=0.05) e sottili
            
            # Un piccolo trucco: disegniamo la label della legenda solo per la PRIMA simulazione
            lbl = f"R{robot_data['id']} Trajectories" if run_idx == 0 else ""
            
            plt.plot(robot_data['x'], robot_data['y'], color=color, alpha=0.1, linewidth=0.5, label=lbl)

    plt.title("Swarm Exploration Coverage (100 Simulations)")
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
    plt.savefig('grafico_traiettorie.png', dpi=300)
    print("Grafico 3: 'grafico_traiettorie.png' salvato.")

    # =========================================================
# GRAFICO 4: Distribuzione delle Scoperte (Pie Chart)
# =========================================================
first_finders = data.get('first_finders', [])

if first_finders:
    # Contiamo quante volte ha vinto R1, R2 e R3
    wins = [
        first_finders.count(1),
        first_finders.count(2),
        first_finders.count(3)
    ]
    
    labels = ['Robot 1 (Red)', 'Robot 2 (Blue)', 'Robot 3 (Green)']
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    plt.figure(figsize=(7, 7))
    # Creiamo il grafico a torta. 'explode' stacca leggermente le fette per un look più moderno
    plt.pie(wins, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=140, explode=(0.05, 0.05, 0.05), shadow=True)
    
    plt.title("First Detection Distribution")
    plt.tight_layout()
    plt.savefig('grafico_scoperte.png', dpi=300)
    print("Grafico 4: 'grafico_scoperte.png' salvato.")

plt.show()