#!/usr/bin/env python3
"""
bin_to_imu_csv.py
------------------
Convertit le fichier binaire .bin (log carte SD Crazyflie, decode via
cfusdlog.py) en un CSV imu_reel.csv utilisable par le pipeline FGO
existant.

IMPORTANT -- conversions d'unites appliquees (le .bin stocke des
unites "brutes" Crazyflie, pas SI) :
  - acc.x/y/z sont en G (multiples de g=9.8 m/s^2) -> converties en m/s^2
  - gyro.x/y/z sont en deg/s -> converties en rad/s (x pi/180)

Le temps (colonne "t") est laisse dans l'horloge NATIVE du Crazyflie,
en SECONDES (pas relativise a zero ici) -- c'est volontaire : le
script mocap_to_csv.py doit produire gps_reel.csv/odom_reel.csv sur
la MEME horloge (via la colonne de sync Crazyflie du CSV mocap), pour
que fgo_real_flight.py puisse aligner les deux automatiquement (il
fait deja `t0 = df_gps["t"].iloc[0]` et soustrait ce t0 a tout,
IMU inclus -- comme pour les donnees simulees).

Usage :
  python3 vol_reel/bin_to_imu_csv.py chemin/vers/log.bin
  -> ecrit csv_files/imu_reel.csv
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import csv
import cfusdlog

GRAVITY = 9.8  # coherent avec le reste du projet (GRAVITY utilise partout ailleurs)

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csv_files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bin_file", help="chemin vers le fichier .bin (log SD card)")
    parser.add_argument("--out", default=os.path.join(CSV_DIR, "imu_reel.csv"))
    args = parser.parse_args()

    print(f"[bin_to_imu_csv] Decodage de {args.bin_file}...")
    log = cfusdlog.decode(args.bin_file)
    data = log["fixedFrequency"]

    keys = list(data.keys())
    print(f"[bin_to_imu_csv] Colonnes disponibles : {keys}")

    required = ["timestamp", "acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]
    missing = [k for k in required if k not in keys]
    if missing:
        print(f"[ERREUR] Colonnes manquantes dans le .bin : {missing}")
        print("         Verifie que la config de log embarquee incluait bien accel+gyro.")
        sys.exit(1)

    # Timestamp du .bin : deja en millisecondes (horloge native Crazyflie).
    # On convertit en secondes, SANS relativiser a zero -- voir docstring.
    t_s = np.asarray(data["timestamp"], dtype=float) / 1000.0

    # ---- Conversions d'unites vers SI ----
    ax = np.asarray(data["acc.x"], dtype=float) * GRAVITY
    ay = np.asarray(data["acc.y"], dtype=float) * GRAVITY
    az = np.asarray(data["acc.z"], dtype=float) * GRAVITY

    wx = np.radians(np.asarray(data["gyro.x"], dtype=float))
    wy = np.radians(np.asarray(data["gyro.y"], dtype=float))
    wz = np.radians(np.asarray(data["gyro.z"], dtype=float))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "ax", "ay", "az", "wx", "wy", "wz"])
        for i in range(len(t_s)):
            w.writerow([f"{t_s[i]:.6f}", f"{ax[i]:.6f}", f"{ay[i]:.6f}", f"{az[i]:.6f}",
                        f"{wx[i]:.6f}", f"{wy[i]:.6f}", f"{wz[i]:.6f}"])

    print(f"[bin_to_imu_csv] {len(t_s)} echantillons ecrits -> {args.out}")
    print(f"[bin_to_imu_csv] Plage temporelle (horloge Crazyflie) : "
          f"{t_s[0]:.3f}s a {t_s[-1]:.3f}s (duree {t_s[-1]-t_s[0]:.1f}s)")
    print(f"[bin_to_imu_csv] Verification unites -- az moyen (devrait etre proche de "
          f"+/-{GRAVITY} si un axe est vertical au repos) : {az[:50].mean():.3f} m/s^2")


if __name__ == "__main__":
    main()
