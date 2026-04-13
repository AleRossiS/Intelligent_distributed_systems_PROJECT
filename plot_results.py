import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. Carica i Dati
with open('experiment_results4.pkl', 'rb') as f:
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

plt.show()