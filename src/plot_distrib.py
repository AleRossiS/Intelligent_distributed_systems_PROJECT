# from motion_and_mapping import MotionController

# ctrl = MotionController()

# u = ctrl.plot_levy_distribution()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Parametri ---
goal_x, goal_y = 0.0, 0.0
sigma_0 = 0.5
R_ad = 5.0
ring_radius = 2.5
dist_to_goal = 1.0  # Messo a 1.0 per avere un anello ben definito
sigma_adaptive = sigma_0 * (1.0 + (dist_to_goal / R_ad))

# --- 2. Griglia ad alta risoluzione ---
grid_res = 300  # Molto più alto per superfici lisce
x = np.linspace(-6, 6, grid_res)
y = np.linspace(-6, 6, grid_res)
X, Y = np.meshgrid(x, y)

dists_to_goal_grid = np.hypot(X - goal_x, Y - goal_y)
weights = np.exp(-(dists_to_goal_grid - ring_radius)**2 / (2 * sigma_adaptive**2))

# --- 3. Stile del Plot ---
# Usiamo un font in stile accademico e forziamo il mathtext per evitare bug con il LaTeX
plt.rc('font', family='serif')
plt.rcParams['mathtext.fontset'] = 'cm' 

fig = plt.figure(figsize=(10, 8)) # Finestra più quadrata, adatta a un singolo plot
fig.patch.set_facecolor('white')

# === GRAFICO 3D (Surface) ===
ax = fig.add_subplot(1, 1, 1, projection='3d') # 1 riga, 1 colonna, indice 1

# Superficie liscia
surf = ax.plot_surface(X, Y, weights, cmap='plasma', 
                        linewidth=0, antialiased=True, alpha=0.8,
                        rstride=2, cstride=2)

# Proiezione dell'ombra sul pavimento (offset negativo)
ax.contourf(X, Y, weights, zdir='z', offset=-0.1, levels=50, cmap='plasma', alpha=0.3)

ax.set_zlim(-0.1, 1.05)

# Assi X e Y
ax.set_xlabel('x [m]', fontsize=12, labelpad=10)
ax.set_ylabel('y [m]', fontsize=12, labelpad=10)

# Asse Z: disabilitiamo la rotazione automatica e impostiamo l'etichetta dritta
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r'$\phi(q)$', fontsize=16, labelpad=10, rotation=0)

# Pulizia dello sfondo del 3D
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Schiarisce la griglia del 3D
ax.xaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 1)
ax.yaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 1)
ax.zaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 1)

# Imposta un'angolazione della telecamera elegante
ax.view_init(elev=35, azim=45)

plt.tight_layout()
plt.show()