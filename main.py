#!/usr/bin/env python3
"""
main.py
-------
Collecte passive des capteurs — drone au sol statique.
Lance la collecte GPS + IMU + Odom pendant DURATION secondes
puis sauvegarde les CSV et génère les plots.

Usage :
    python3 main.py
"""

import time
import matplotlib.pyplot as plt

from utils.collect import init_nodes, spin_once, stop_nodes, save_all
from plot_scripts.plot_sensors import fig1_gps_vs_odom, fig2_imu_accel, fig3_imu_gyro

# ─── Paramètre modifiable ─────────────────────────────────────────────────────
DURATION = 30  # durée de collecte en secondes
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Initialisation ROS2 + nodes capteurs
    executor, gps_node, imu_node, odom_node = init_nodes()

    # Boucle de collecte
    print(f"[Main] Collecte en cours pendant {DURATION} secondes...")
    t_start = time.time()
    while time.time() - t_start < DURATION:   
        spin_once(executor)

    # Arrêt ROS2
    stop_nodes(executor, gps_node, imu_node, odom_node)

    # Sauvegarde CSV
    save_all("sol", gps_node, imu_node, odom_node)

    # Plots
    fig1_gps_vs_odom("csv_files/gps.csv", "csv_files/odom.csv", titre="sol")
    fig2_imu_accel("csv_files/imu.csv", "csv_files/odom.csv", statique=True, titre="sol")
    fig3_imu_gyro("csv_files/imu.csv", "csv_files/odom.csv", statique=True, titre="sol")
    plt.show()


if __name__ == "__main__":
    main()