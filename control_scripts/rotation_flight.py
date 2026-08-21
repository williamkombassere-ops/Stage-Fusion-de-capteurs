#!/usr/bin/env python3
"""
rotation_flight.py
------------------
Séquence simple :
  1. Monte à Z_CIBLE
  2. Descend jusqu'au sol EN TOURNANT sur le yaw (rotation continue pendant
     toute la descente, pas de phase de stabilisation séparée)

Usage : python3 control_scripts/rotation_flight.py
CSV   : csv_files/gps_rotation.csv / imu_rotation.csv / odom_rotation.csv
"""

import sys, os, signal, time, math, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import rclpy
from geometry_msgs.msg import Twist
from utils.collect import init_nodes, spin_once, stop_nodes, save_all

# ─── Paramètres ───────────────────────────────────────────────────────────────
Z_CIBLE = 1.0    # altitude cible avant la descente (m)
VZ      = 0.15   # vitesse montée/descente (m/s)
W_YAW   = 0.5    # vitesse angulaire yaw pendant la descente (rad/s)
NOM     = "rotation"
TOPIC   = "/crazyflie/cmd_vel"
# ─────────────────────────────────────────────────────────────────────────────

_exec = _gps = _imu = _odom = _ctrl = _pub = None


def twist(vz=0.0, wz=0.0):
    """Publie vz (vertical) + wz (yaw)."""
    msg = Twist()
    msg.linear.z  = vz
    msg.angular.z = wz
    _pub.publish(msg)


def z():
    """Altitude courante (Odom)."""
    return _odom.data[-1]["z"] if _odom.data else 0.0


def yaw():
    """Yaw courant en degrés (quaternion Odom)."""
    if not _odom.data:
        return 0.0
    d = _odom.data[-1]
    return math.degrees(math.atan2(
        2*(d["qw"]*d["qz"] + d["qx"]*d["qy"]),
        1 - 2*(d["qy"]**2 + d["qz"]**2)))


def spin():
    """Spin ROS2."""
    spin_once(_exec, timeout_sec=0.05)


def shutdown(sig=None, frame=None):
    """Arrêt propre + sauvegarde CSV."""
    print("\n[Arrêt] Sauvegarde CSV...")
    twist()
    stop_nodes(_exec, _gps, _imu, _odom, _ctrl)
    save_all(NOM, _gps, _imu, _odom)
    sys.exit(0)


def main():
    global _exec, _gps, _imu, _odom, _ctrl, _pub
    signal.signal(signal.SIGINT, shutdown)

    _exec, _gps, _imu, _odom = init_nodes()
    _ctrl = rclpy.create_node("rotation_ctrl")
    _pub  = _ctrl.create_publisher(Twist, TOPIC, 10)
    _exec.add_node(_ctrl)

    # ── Phase 1 : Décollage ───────────────────────────────────────────────
    print(f"[1] Décollage vers {Z_CIBLE}m...")
    t0 = time.time()
    while z() < Z_CIBLE - 0.02 and time.time()-t0 < 30:
        twist(vz=VZ); spin()
        print(f"  z={z():.3f}m", end="\r")
    twist()
    print(f"\n[1] Altitude atteinte : z={z():.3f}m")

    # ── Phase 2 : Descente en tournant sur le yaw ─────────────────────────
    print(f"[2] Descente en rotation (yaw continu) jusqu'au sol...")
    t0 = time.time()
    while z() > 0.02 and time.time()-t0 < 30:
        twist(vz=-VZ, wz=W_YAW)  # descente + rotation yaw simultanées
        spin()
        print(f"  yaw={yaw():.1f}°  z={z():.3f}m", end="\r")
    twist()
    print(f"\n[2] Drone au sol : yaw={yaw():.1f}°  z={z():.3f}m")

    shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERREUR] {e}")
        traceback.print_exc()
        shutdown()