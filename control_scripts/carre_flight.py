#!/usr/bin/env python3
"""
carre_flight.py
--------------------
Vol en carre de 2m de cote, a altitude constante de 1m.

Sequence :
  1. Decollage jusqu'a Z_CIBLE = 1.0m
  2. Vol en carre RAPIDE, cote = 2.0m, a altitude constante : le drone
     vise chaque coin l'un apres l'autre (control par waypoints, pas
     par courbe parametrique continue -- plus robuste).
  3. Atterrissage

Control : a chaque instant, commande de vitesse proportionnelle a
l'ecart entre la position reelle (Odom) et le coin cible courant.
L'altitude est corrigee EN PERMANENCE (pas seulement au decollage) --
meme pendant les phases de deplacement horizontal, pour eviter toute
derive verticale progressive durant le carre.

ROBUSTESSE (par rapport a une version simple) :
  - TIMEOUT_COIN : si un coin n'est jamais atteint (ex: derive GPS/IMU,
    saturation moteur), on passe au coin suivant apres ce delai au lieu
    de bloquer le vol indefiniment.
  - Correction d'altitude active a CHAQUE iteration de la boucle
    principale (meme logique que le decollage/atterrissage), pas
    seulement pendant les phases dediees.
  - Verification de connexion Odom avant de commencer (evite de
    commander le drone "a l'aveugle" si aucune donnee de position
    n'est encore recue).
  - Log de progression clair (coin courant, distance, altitude, temps
    ecoule) pour diagnostiquer un vol en cours si besoin.

Usage : python3 control_scripts/carre_flight.py
CSV   : csv_files/gps_carre.csv / imu_carre.csv / odom_carre.csv
"""

import sys, os, signal, time, math, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import rclpy
from geometry_msgs.msg import Twist
from utils.collect import init_nodes, spin_once, stop_nodes, save_all

# ─── Parametres ───────────────────────────────────────────────────────────────
Z_CIBLE       = 1.0    # altitude cible (m)
VZ            = 0.15   # vitesse de montee/descente (m/s)
COTE          = 2.0    # cote du carre (m)

KP_XY         = 1.2    # gain P pour suivre les waypoints
KP_Z          = 0.8    # gain P pour maintenir l'altitude constante
V_MAX         = 0.8    # saturation vitesse horizontale (m/s)
VZ_MAX        = 0.4    # saturation vitesse verticale pendant le carre (m/s)
                        # (plus petit que VZ pour eviter les a-coups
                        # verticaux pendant un deplacement horizontal rapide)

SEUIL_COIN    = 0.15   # distance (m) sous laquelle on considere le coin atteint
TIMEOUT_COIN  = 25.0   # secondes max passees a viser un seul coin avant
                        # d'abandonner et de passer au suivant (securite
                        # contre un blocage indefini)
TIMEOUT_DECOLLAGE   = 30.0  # secondes max pour atteindre Z_CIBLE
TIMEOUT_ATTERRISSAGE = 30.0  # secondes max pour redescendre au sol
TIMEOUT_ATTENTE_ODOM = 10.0  # secondes max pour attendre la 1ere donnee Odom

NOM         = "carre2m"  # nom utilise pour les CSV -- EXPLICITE sur la taille
                        # (2m de cote) pour ne jamais ecraser un autre vol carre
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
        return None
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


def attendre_odom():
    print("[0] Attente des premieres donnees Odom...")
    t0 = time.time()
    while position() is None:
        spin()
        if time.time() - t0 > TIMEOUT_ATTENTE_ODOM:
            raise RuntimeError(
                f"Aucune donnee Odom recue apres {TIMEOUT_ATTENTE_ODOM}s -- "
                f"verifier que la simulation/le drone publie bien sur Odom.")
    print(f"[0] Odom recu : position initiale = {position()}")


def main():
    global _exec, _gps, _imu, _odom, _ctrl, _pub
    signal.signal(signal.SIGINT, shutdown)

    _exec, _gps, _imu, _odom = init_nodes()
    _ctrl = rclpy.create_node("carre_flight_ctrl")
    _pub  = _ctrl.create_publisher(Twist, TOPIC, 10)
    _exec.add_node(_ctrl)

    attendre_odom()

    # ── Phase 1 : Decollage ───────────────────────────────────────────────
    print(f"[1] Decollage vers {Z_CIBLE}m...")
    t0 = time.time()
    while position()[2] < Z_CIBLE - 0.02:
        if time.time() - t0 > TIMEOUT_DECOLLAGE:
            print(f"\n[1] ATTENTION : decollage incomplet apres {TIMEOUT_DECOLLAGE}s "
                  f"(z={position()[2]:.3f}m) -- on continue quand meme.")
            break
        twist(vz=VZ)
        spin()
        print(f"  z={position()[2]:.3f}m", end="\r")
    twist()
    print(f"\n[1] Altitude atteinte : z={position()[2]:.3f}m")

    # ── Phase 2 : carre par waypoints, altitude corrigee en permanence ─────
    x0, y0, _ = position()
    waypoints = [
        (x0 + COTE, y0),
        (x0 + COTE, y0 + COTE),
        (x0,        y0 + COTE),
        (x0,        y0),
    ]
    print(f"[2] Debut du carre {COTE}m, depart=({x0:.3f}, {y0:.3f}), altitude cible={Z_CIBLE}m")

    t_debut = time.time()
    for i, (xc, yc) in enumerate(waypoints):
        print(f"[2] Vers coin {i+1}/4 : ({xc:.2f}, {yc:.2f})")
        t_coin_debut = time.time()
        while True:
            x_reel, y_reel, z_reel = position()
            dx, dy = xc - x_reel, yc - y_reel
            dist = math.hypot(dx, dy)

            if dist < SEUIL_COIN:
                break
            if time.time() - t_coin_debut > TIMEOUT_COIN:
                print(f"\n[2] ATTENTION : coin {i+1} non atteint apres {TIMEOUT_COIN}s "
                      f"(dist={dist:.2f}m restants) -- passage au coin suivant.")
                break

            vx = saturer(KP_XY * dx, V_MAX)
            vy = saturer(KP_XY * dy, V_MAX)
            # Correction d'altitude active en PERMANENCE pendant le carre,
            # pas seulement au decollage/atterrissage -- evite toute derive
            # verticale progressive durant les 4 segments.
            vz = saturer(KP_Z * (Z_CIBLE - z_reel), VZ_MAX)

            twist(vx=vx, vy=vy, vz=vz)
            spin()
            t_ecoule = time.time() - t_debut
            print(f"  t={t_ecoule:5.1f}s  x={x_reel:+.2f} y={y_reel:+.2f} z={z_reel:.2f}  "
                  f"dist_coin={dist:.2f}m", end="\r")

    twist()
    duree_carre = time.time() - t_debut
    x_fin, y_fin, z_fin = position()
    print(f"\n[2] Carre termine en {duree_carre:.1f}s : x={x_fin:.3f} y={y_fin:.3f} z={z_fin:.3f}")

    # ── Phase 3 : Atterrissage ──────────────────────────────────────────────
    print("[3] Atterrissage...")
    t0 = time.time()
    while position()[2] > 0.02:
        if time.time() - t0 > TIMEOUT_ATTERRISSAGE:
            print(f"\n[3] ATTENTION : atterrissage incomplet apres {TIMEOUT_ATTERRISSAGE}s "
                  f"(z={position()[2]:.3f}m) -- arret force.")
            break
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