#!/usr/bin/env python3
"""
triangle_flight.py
--------------------
Sequence :
  1. Decollage jusqu'a Z_CIBLE = 0.5m
  2. Vol en triangle ISOCELE, a altitude constante : le drone vise
     chaque sommet l'un apres l'autre (control par waypoints, meme
     logique que square2m_flight.py -- robuste, evite le probleme
     rencontre avec le "8" ou le suivi de courbe theorique n'etait pas
     assez precis).
     Geometrie : 2 cotes egaux de 2.0m (COTE_EGAL), base de 1.0m (BASE).
     Sommets calcules automatiquement par trigonometrie (hauteur du
     triangle isocele) a partir du point de depart :
       A = (x0, y0)                              -- premier sommet de base
       B = (x0 + BASE, y0)                        -- second sommet de base
       C = (x0 + BASE/2, y0 + HAUTEUR)             -- sommet oppose (apex)
     avec HAUTEUR = sqrt(COTE_EGAL^2 - (BASE/2)^2) (Pythagore).
  3. Atterrissage

Control : identique a square2m_flight.py -- a chaque instant, commande
de vitesse proportionnelle a l'ecart entre la position reelle (Odom) et
le sommet cible courant. Passage au sommet suivant des que la distance
au sommet cible passe sous un seuil (SEUIL_SOMMET).

Usage : python3 control_scripts/triangle_flight.py
CSV   : csv_files/gps_triangle.csv / imu_triangle.csv / odom_triangle.csv
"""

import sys, os, signal, time, math, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import rclpy
from geometry_msgs.msg import Twist
from utils.collect import init_nodes, spin_once, stop_nodes, save_all

# ─── Parametres ───────────────────────────────────────────────────────────────
Z_CIBLE     = 0.5    # altitude cible (m) -- vol bas, comme demande
VZ          = 0.15   # vitesse de montee/descente (m/s)

COTE_EGAL   = 2.0    # longueur des 2 cotes egaux du triangle isocele (m)
BASE        = 2.0    # longueur de la base (m)
HAUTEUR     = math.sqrt(COTE_EGAL**2 - (BASE / 2.0)**2)  # Pythagore -- hauteur
                      # du triangle isocele, depuis le milieu de la base
                      # jusqu'au sommet oppose. Avec COTE_EGAL=2.0, BASE=2.0 :
                      # HAUTEUR = sqrt(4 - 1) = sqrt(3) ~= 1.7321 m
                      # (triangle equilateral dans ce cas precis)

KP_XY       = 1.2    # gain P pour suivre les waypoints (identique au carre)
KP_Z        = 0.8    # gain P pour maintenir l'altitude constante
V_MAX       = 0.8    # saturation vitesse horizontale (m/s)
SEUIL_SOMMET = 0.15  # distance (m) sous laquelle on considere le sommet
                      # atteint et on passe au suivant

NOM         = "triangle"
TOPIC       = "/crazyflie/cmd_vel"
# ─────────────────────────────────────────────────────────────────────────────

_exec = _gps = _imu = _odom = _ctrl = _pub = None


def twist(vx=0.0, vy=0.0, vz=0.0, wz=0.0):
    msg = Twist()
    msg.linear.x  = vx
    msg.linear.y  = vy
    msg.linear.z  = vz
    msg.angular.z = wz
    _pub.publish(msg)


def position():
    if not _odom.data:
        return 0.0, 0.0, 0.0
    d = _odom.data[-1]
    return d["x"], d["y"], d["z"]


def saturer(v: float, v_max: float) -> float:
    return max(-v_max, min(v_max, v))


def spin():
    spin_once(_exec, timeout_sec=0.05)


def shutdown(sig=None, frame=None):
    print("\n[Arret] Sauvegarde CSV...")
    twist()
    stop_nodes(_exec, _gps, _imu, _odom, _ctrl)
    save_all(NOM, _gps, _imu, _odom)
    sys.exit(0)


def main():
    global _exec, _gps, _imu, _odom, _ctrl, _pub
    signal.signal(signal.SIGINT, shutdown)

    _exec, _gps, _imu, _odom = init_nodes()
    _ctrl = rclpy.create_node("triangle_ctrl")
    _pub  = _ctrl.create_publisher(Twist, TOPIC, 10)
    _exec.add_node(_ctrl)

    # ── Phase 1 : Decollage ───────────────────────────────────────────────
    print(f"[1] Decollage vers {Z_CIBLE}m...")
    t0 = time.time()
    while position()[2] < Z_CIBLE - 0.02 and time.time() - t0 < 30:
        twist(vz=VZ)
        spin()
        print(f"  z={position()[2]:.3f}m", end="\r")
    twist()
    print(f"\n[1] Altitude atteinte : z={position()[2]:.3f}m")

    # ── Phase 2 : triangle isocele par waypoints ────────────────────────────
    x0, y0, _ = position()
    waypoints = [
        (x0 + BASE,       y0),                 # B : second sommet de base
        (x0 + BASE / 2.0, y0 + HAUTEUR),        # C : apex (sommet oppose)
        (x0,               y0),                # retour a A : ferme la boucle
    ]
    print(f"[2] Debut du triangle isocele (cotes={COTE_EGAL}m, base={BASE}m, "
          f"hauteur={HAUTEUR:.4f}m), depart=({x0:.3f}, {y0:.3f})")

    t_debut = time.time()
    for i, (xc, yc) in enumerate(waypoints):
        print(f"[2] Vers sommet {i+1}/3 : ({xc:.2f}, {yc:.2f})")
        while True:
            x_reel, y_reel, z_reel = position()
            dx, dy = xc - x_reel, yc - y_reel
            dist = math.hypot(dx, dy)
            if dist < SEUIL_SOMMET:
                break
            vx = saturer(KP_XY * dx, V_MAX)
            vy = saturer(KP_XY * dy, V_MAX)
            vz = saturer(KP_Z * (Z_CIBLE - z_reel), VZ)
            twist(vx=vx, vy=vy, vz=vz)
            spin()
            print(f"  x={x_reel:+.2f} y={y_reel:+.2f} z={z_reel:.2f}  dist_sommet={dist:.2f}m", end="\r")

    twist()
    duree_triangle = time.time() - t_debut
    x_fin, y_fin, z_fin = position()
    print(f"\n[2] Triangle termine en {duree_triangle:.1f}s : x={x_fin:.3f} y={y_fin:.3f} z={z_fin:.3f}")

    # ── Phase 3 : Atterrissage ──────────────────────────────────────────────
    print("[3] Atterrissage...")
    t0 = time.time()
    while position()[2] > 0.02 and time.time() - t0 < 30:
        twist(vz=-VZ)
        spin()
        print(f"  z={position()[2]:.3f}m", end="\r")
    twist()
    print(f"\n[3] Drone au sol : z={position()[2]:.3f}m")

    shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERREUR] {e}")
        traceback.print_exc()
        shutdown()