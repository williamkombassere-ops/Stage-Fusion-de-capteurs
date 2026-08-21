#!/usr/bin/env python3
"""
vertical_flight.py
------------------
Fait monter le drone à Z_CIBLE, maintient l'altitude, puis redescend. Collecte GPS + IMU + Odom + Flow Deck (flux optique + ToF) en parallèle via utils/collect.py.
Usage :
    python3 control_scripts/vertical_flight.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import traceback
import matplotlib.pyplot as plt
import signal
import rclpy
from geometry_msgs.msg import Twist
from utils.collect import init_nodes, spin_once, stop_nodes, save_all
from data_read.read_flow import FlowReader, TofReader
from plot_scripts.plot_sensors import fig1_gps_vs_odom, fig2_imu_accel, fig3_imu_gyro

# ─── Paramètres modifiables ───────────────────────────────────────────────────
Z_CIBLE   = 1.0   # altitude cible (m)
VZ        = 0.15  # vitesse verticale (m/s)
HOLD_TIME = 1.0   # durée de maintien à l'altitude cible (s)
KP_Z      = 1.5   # gain correcteur altitude
NOM       = "vertical"
# ──────────────────────────────────────────────────────────────────────────────

TOPIC_CMD = "/crazyflie/cmd_vel"
_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(_ROOT, "csv_files")

# Variables globales pour SIGINT
_exec = _gps = _imu = _odom = _flow = _tof = _ctrl = _pub_g = None

def _sigint(sig=None, frame=None):
    print("\n[Arrêt] Ctrl+C — sauvegarde CSV...")
    try:
        if _pub_g:
            send_twist(_pub_g)
        if _exec:
            stop_nodes(_exec, _gps, _imu, _odom, _flow, _tof, _ctrl)
        if _gps:
            save_all("vertical", _gps, _imu, _odom, flow_node=_flow, tof_node=_tof)
    except: pass
    import sys; sys.exit(0)

signal.signal(signal.SIGINT, _sigint)

def send_twist(pub, vx=0.0, vy=0.0, vz=0.0):
    """Publie une commande Twist."""
    msg = Twist()
    msg.linear.x, msg.linear.y, msg.linear.z = vx, vy, vz
    pub.publish(msg)

def get_z(odom_node):
    """Retourne la dernière altitude Odom."""
    return odom_node.data[-1]["z"] if odom_node.data else 0.0

def main():
    global _exec, _gps, _imu, _odom, _flow, _tof, _ctrl, _pub_g

    # Init ROS2 (rclpy.init() est appelé à l'intérieur de init_nodes) +
    # capteurs standards (gps, imu, odom)
    executor, gps_node, imu_node, odom_node = init_nodes()

    # Flow Deck : créés APRÈS init_nodes(), car rclpy.init() doit avoir
    # été appelé avant de pouvoir créer un Node (sinon NotInitializedException)
    flow_node = FlowReader()
    tof_node  = TofReader()
    executor.add_node(flow_node)
    executor.add_node(tof_node)

    # Ajout du node de contrôle
    ctrl_node = rclpy.create_node("vertical_controller")
    pub = ctrl_node.create_publisher(Twist, TOPIC_CMD, 10)
    executor.add_node(ctrl_node)

    _exec, _gps, _imu, _odom = executor, gps_node, imu_node, odom_node
    _flow, _tof, _ctrl, _pub_g = flow_node, tof_node, ctrl_node, pub

    # ── Phase 1 : Décollage ───────────────────────────────────────────────
    print(f"[Phase 1] Décollage vers z={Z_CIBLE}m...")
    while get_z(odom_node) < Z_CIBLE - 0.02:
        send_twist(pub, vz=VZ)
        spin_once(executor)
        print(f"  z={get_z(odom_node):.3f}m", end="\r")
    send_twist(pub)
    print(f"\n[Phase 1] Altitude atteinte : z={get_z(odom_node):.3f}m")

    # ── Phase 2 : Maintien ────────────────────────────────────────────────
    print(f"[Phase 2] Maintien à {Z_CIBLE}m pendant {HOLD_TIME}s...")
    t_hold = time.time()
    while time.time() - t_hold < HOLD_TIME:
        vz_corr = max(-0.1, min(0.1, KP_Z * (Z_CIBLE - get_z(odom_node))))
        send_twist(pub, vz=vz_corr)
        spin_once(executor)
    send_twist(pub)

    # ── Phase 3 : Atterrissage ────────────────────────────────────────────
    print("[Phase 3] Atterrissage...")
    t_att = time.time()
    while get_z(odom_node) > 0.02 and time.time() - t_att < 30.0:
        send_twist(pub, vz=-VZ)
        spin_once(executor)
        print(f"  z={get_z(odom_node):.3f}m", end="\r")
    send_twist(pub)
    print(f"\n[Phase 3] Drone au sol.")

    # Arrêt ROS2
    stop_nodes(executor, gps_node, imu_node, odom_node, flow_node, tof_node, ctrl_node)

    # Sauvegarde CSV (gps, imu, odom, flow, tof)
    save_all(NOM, gps_node, imu_node, odom_node, flow_node=flow_node, tof_node=tof_node)

    # Plots — chemins absolus (voir CSV_DIR en haut du fichier)
    fig1_gps_vs_odom(os.path.join(CSV_DIR, f"gps_{NOM}.csv"),
                     os.path.join(CSV_DIR, f"odom_{NOM}.csv"), titre=NOM)
    fig2_imu_accel(os.path.join(CSV_DIR, f"imu_{NOM}.csv"),
                   os.path.join(CSV_DIR, f"odom_{NOM}.csv"), statique=False, titre=NOM)
    fig3_imu_gyro(os.path.join(CSV_DIR, f"imu_{NOM}.csv"),
                  os.path.join(CSV_DIR, f"odom_{NOM}.csv"), statique=False, titre=NOM)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERREUR] {e}")
        traceback.print_exc()