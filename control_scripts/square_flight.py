#!/usr/bin/env python3
"""
square_flight.py
----------------
Fait voler le drone en carré de 1m de côté à 1m d'altitude.
Collecte GPS + IMU + Odom + Flow Deck (flux optique + ToF) via utils/collect.py.

Séquence : décollage → (1,0) → (1,1) → (0,1) → (0,0) → atterrissage

Usage :
    python3 control_scripts/square_flight.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time, math, traceback
import matplotlib.pyplot as plt

import signal
import rclpy
from geometry_msgs.msg import Twist

from utils.collect import init_nodes, spin_once, stop_nodes, save_all
from data_read.read_flow import FlowReader, TofReader
from plot_scripts.plot_sensors import fig1_gps_vs_odom, fig2_imu_accel, fig3_imu_gyro


COTE      = 1.0
Z_CIBLE   = 1.0
VZ        = 0.15
VXY       = 0.15
T_PAUSE   = 1.0
KP_Z      = 1.5
KP_XY     = 1.5
NOM       = "carre"
COINS     = [(COTE, 0.0), (COTE, COTE), (0.0, COTE), (0.0, 0.0)]
TOPIC_CMD = "/crazyflie/cmd_vel"


# Chemin absolu vers csv_files/, ancré sur la racine du projet
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
            save_all("carre", _gps, _imu, _odom, flow_node=_flow, tof_node=_tof)
    except: pass
    import sys; sys.exit(0)

signal.signal(signal.SIGINT, _sigint)

def send_twist(pub, vx=0.0, vy=0.0, vz=0.0):
    msg = Twist()
    msg.linear.x, msg.linear.y, msg.linear.z = vx, vy, vz
    pub.publish(msg)


def get_pos(odom_node):
    if odom_node.data:
        d = odom_node.data[-1]
        return d["x"], d["y"], d["z"]
    return 0.0, 0.0, 0.0


def voler_vers(pub, executor, odom_node, xc, yc):
    """Vole vers (xc, yc) avec correcteur P saturé, maintient l'altitude."""
    while True:
        x, y, z = get_pos(odom_node)
        ex, ey  = xc - x, yc - y
        dist    = math.sqrt(ex**2 + ey**2)
        if dist < 0.05:
            break
        norm = max(dist, 1e-6)
        vx   = max(-VXY, min(VXY, KP_XY * ex / norm * dist))
        vy   = max(-VXY, min(VXY, KP_XY * ey / norm * dist))
        vz   = max(-0.2, min(0.2, KP_Z * (Z_CIBLE - z)))
        send_twist(pub, vx=vx, vy=vy, vz=vz)
        spin_once(executor)
    send_twist(pub)


def main():
    global _exec, _gps, _imu, _odom, _flow, _tof, _ctrl, _pub_g

    # Init ROS2 (rclpy.init() appelé à l'intérieur de init_nodes) +
    # capteurs standards (gps, imu, odom)
    executor, gps_node, imu_node, odom_node = init_nodes()

    # Flow Deck : créés APRÈS init_nodes(), car rclpy.init() doit avoir
    # été appelé avant de pouvoir créer un Node
    flow_node = FlowReader()
    tof_node  = TofReader()
    executor.add_node(flow_node)
    executor.add_node(tof_node)

    ctrl_node = rclpy.create_node("square_controller")
    pub = ctrl_node.create_publisher(Twist, TOPIC_CMD, 10)
    executor.add_node(ctrl_node)

    _exec, _gps, _imu, _odom = executor, gps_node, imu_node, odom_node
    _flow, _tof, _ctrl, _pub_g = flow_node, tof_node, ctrl_node, pub

    # Décollage
    print(f"[Phase 1] Décollage vers z={Z_CIBLE}m...")
    while get_pos(odom_node)[2] < Z_CIBLE - 0.02:
        send_twist(pub, vz=VZ)
        spin_once(executor)
    send_twist(pub)
    print(f"[Phase 1] Altitude : z={get_pos(odom_node)[2]:.3f}m")

    # Carré
    labels = ["(1,0)", "(1,1)", "(0,1)", "(0,0)"]
    for (xc, yc), label in zip(COINS, labels):
        print(f"[Phase 2] Vol vers {label}...")
        voler_vers(pub, executor, odom_node, xc, yc)
        t0 = time.time()
        while time.time() - t0 < T_PAUSE:
            x, y, z = get_pos(odom_node)
            send_twist(pub, vz=max(-0.1, min(0.1, KP_Z * (Z_CIBLE - z))))
            spin_once(executor)

    send_twist(pub)
    print("[Phase 2] Carré terminé.")

    # Atterrissage
    print("[Phase 3] Atterrissage...")
    t_att = time.time()
    while get_pos(odom_node)[2] > 0.02 and time.time() - t_att < 30.0:
        send_twist(pub, vz=-VZ)
        spin_once(executor)
    send_twist(pub)
    print("[Phase 3] Drone au sol.")

    stop_nodes(executor, gps_node, imu_node, odom_node, flow_node, tof_node, ctrl_node)
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