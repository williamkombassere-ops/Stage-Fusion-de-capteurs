#!/usr/bin/env python3
"""
mocap_to_csv.py
-----------------
Convertit le CSV mocap (QTM), qui contient la position verite-terrain
(x,y,z) du vol reel, en DEUX fichiers pour le pipeline FGO :

  1. odom_reel.csv (t, x, y, z)        -- la verite (mocap brute)
  2. gps_reel.csv  (t, x, y, z)        -- un "GPS" SYNTHETIQUE : la
     meme position mocap + un bruit gaussien ajoute artificiellement
     (sigma=SIGMA_GPS, coherent avec le reste du projet, 1.0m), pour
     simuler ce qu'un vrai GPS aurait mesure. Necessaire car il n'y a
     pas eu de vrai GPS sur ce vol (vol interieur, mocap seulement).

IMPORTANT -- synchronisation temporelle :
  Le CSV mocap a sa PROPRE horloge (QTM), differente de celle du
  Crazyflie (utilisee par bin_to_imu_csv.py). Pour aligner les deux,
  ce script utilise la colonne de timestamp CRAZYFLIE presente dans
  le CSV mocap (--sync-col), PAS la colonne "Timestamp QTM". Les deux
  fichiers de sortie (gps_reel.csv, odom_reel.csv) sont donc, comme
  imu_reel.csv, sur l'horloge Crazyflie native en secondes (pas
  relativisee -- fgo_real_flight.py s'en charge automatiquement).

  Si tu ne connais pas le nom exact de la colonne de sync, lance ce
  script une premiere fois SANS --sync-col : il affichera la liste des
  colonnes disponibles, sans rien ecrire.

Usage :
  python3 vol_reel/mocap_to_csv.py chemin/vers/mocap.csv --sync-col "Timestamp Crazyflie"
"""

import sys, os, argparse
import numpy as np
import pandas as pd
import csv

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csv_files")
SIGMA_GPS = 1.0  # m -- coherent avec SIGMA_GPS utilise partout ailleurs dans le projet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mocap_csv", help="chemin vers le CSV mocap (QTM)")
    parser.add_argument("--sync-col", default=None,
                         help="nom exact de la colonne de timestamp Crazyflie "
                              "(horloge partagee avec le .bin). Obligatoire pour "
                              "generer les fichiers -- lance sans cet argument "
                              "pour juste lister les colonnes disponibles.")
    parser.add_argument("--seed", type=int, default=42,
                         help="graine aleatoire pour le bruit GPS synthetique "
                              "(reproductibilite)")
    args = parser.parse_args()

    df = pd.read_csv(args.mocap_csv)
    print(f"[mocap_to_csv] {len(df)} lignes chargees depuis {args.mocap_csv}")
    print(f"[mocap_to_csv] Colonnes disponibles :")
    for c in df.columns:
        print(f"    - {c!r}")

    if args.sync_col is None:
        print("\n[mocap_to_csv] Aucune --sync-col fournie : arret ici. "
              "Identifie la colonne de timestamp Crazyflie ci-dessus et relance avec "
              "--sync-col \"<nom exact>\"")
        return

    if args.sync_col not in df.columns:
        print(f"[ERREUR] Colonne '{args.sync_col}' introuvable. Voir la liste ci-dessus.")
        sys.exit(1)

    t_s = df[args.sync_col].to_numpy(dtype=float)
    x_raw = df["x"].to_numpy(dtype=float)
    y_raw = df["y"].to_numpy(dtype=float)
    z_raw = df["z"].to_numpy(dtype=float)

    # ── Filtrer les lignes incompletes (NaN dans le sync-col ou x/y/z) ──────
    # Certains CSV mocap ont des lignes de fin/entete partiellement vides.
    valid = ~(np.isnan(t_s) | np.isnan(x_raw) | np.isnan(y_raw) | np.isnan(z_raw))
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"[mocap_to_csv] {n_dropped} ligne(s) avec valeur(s) manquante(s) "
              f"ignoree(s) (sur {len(t_s)} au total)")
    t_s = t_s[valid]
    x_raw = x_raw[valid]
    y_raw = y_raw[valid]
    z_raw = z_raw[valid]

    if len(t_s) == 0:
        print("[ERREUR] Plus aucune ligne valide apres filtrage des NaN -- "
              "verifie le fichier CSV mocap et la colonne --sync-col choisie.")
        sys.exit(1)

    # Si la colonne de sync est en ms (valeurs > 1e5 typiquement pour un
    # timestamp Crazyflie), on convertit en secondes -- heuristique simple.
    # nanmedian (pas median) pour ignorer d'eventuels NaN residuels.
    if np.nanmedian(t_s) > 1e5:
        print("[mocap_to_csv] Valeurs de sync semblent etre en ms -> conversion en s")
        t_s = t_s / 1000.0

    x = x_raw
    y = y_raw
    z = z_raw

    # ---- odom_reel.csv : verite terrain brute (mocap) ----
    out_odom = os.path.join(CSV_DIR, "odom_reel.csv")
    os.makedirs(CSV_DIR, exist_ok=True)
    with open(out_odom, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z"])
        for i in range(len(t_s)):
            w.writerow([f"{t_s[i]:.6f}", f"{x[i]:.6f}", f"{y[i]:.6f}", f"{z[i]:.6f}"])
    print(f"[mocap_to_csv] {len(t_s)} lignes -> {out_odom} (verite terrain)")

    # ---- gps_reel.csv : mocap + bruit gaussien synthetique ----
    rng = np.random.default_rng(args.seed)
    x_bruite = x + rng.normal(0, SIGMA_GPS, size=len(x))
    y_bruite = y + rng.normal(0, SIGMA_GPS, size=len(y))
    z_bruite = z + rng.normal(0, SIGMA_GPS, size=len(z))

    out_gps = os.path.join(CSV_DIR, "gps_reel.csv")
    with open(out_gps, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z"])
        for i in range(len(t_s)):
            w.writerow([f"{t_s[i]:.6f}", f"{x_bruite[i]:.6f}", f"{y_bruite[i]:.6f}", f"{z_bruite[i]:.6f}"])
    print(f"[mocap_to_csv] {len(t_s)} lignes -> {out_gps} "
          f"(GPS synthetique, sigma={SIGMA_GPS}m)")

    print(f"\n[mocap_to_csv] Plage temporelle (horloge Crazyflie, meme que l'IMU attendue) : "
          f"{t_s[0]:.3f}s a {t_s[-1]:.3f}s (duree {t_s[-1]-t_s[0]:.1f}s)")
    print("[mocap_to_csv] VERIFIE que cette plage recouvre bien celle affichee par "
          "bin_to_imu_csv.py -- sinon les fichiers ne correspondent pas au meme vol/segment.")


if __name__ == "__main__":
    main()
