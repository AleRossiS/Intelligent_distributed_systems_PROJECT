# Swarm Intelligence & Target Search

This project simulates a multi-agent robotic swarm designed to explore a 2D environment, detect a hidden target ("treasure"), and coordinate a formation around it. The system integrates swarm intelligence, Voronoi-based coordination, EKF-based localization, and consensus to estimate the treasure position.

## Project Structure

The project is organized into a main execution script and a dedicated `modules/` folder for core logic.

### Core Modules (`modules/`)
- **`robot_class.py`**: Defines the robot agent, managing its physical state, internal EKF estimation, and sensor data processing.
- **`environment_class.py`**: Manages the 2D simulation space, including obstacle definitions, target placement, and Pygame-based rendering.
- **`voronoi_cells.py`**: Computes dynamic Voronoi partitions to ensure safe spatial distribution and coordinated movement.
- **`motion_and_mapping.py`**: Implements the control FSM, including Lévy Flight for exploration, obstacle avoidance, and Lloyd's algorithm for formation.
- **`broadcast.py`**: Simulates the communication network, allowing robots to share state information and reach consensus on target location.
- **`ekf_localization.py`**: Contains the Extended Kalman Filter (EKF) logic and the Global Camera model for noisy state observation.

### Configuration
- **`config.py`**: The centralized hub for all simulation parameters. You can easily modify noise levels (EKF/Camera), sensor ranges, and physical constants without touching the core logic.

## How to Run

1. **Install Dependencies**:
   Ensure you have the required libraries installed:
   ```bash
   pip install numpy pygame scipy shapely matplotlib

2. Run the **`main_test.py`** script.