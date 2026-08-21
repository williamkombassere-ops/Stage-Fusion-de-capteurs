#!/usr/bin/env python3
"""
plot.py
-------
Script de plot universel — fonctionne pour toutes les trajectoires.

Usage :
  python3 plot_scripts/plot.py <trajectoire> [--fgo gps|gps_imu|all]

Exemples :
  python3 plot_scripts/plot.py sol
  python3 plot_scripts/plot.py vertical
  python3 plot_scripts/plot.py vertical --fgo gps
  python3 plot_scripts/plot.py vertical --fgo gps_imu
  python3 plot_scripts/plot.py vertical --fgo all
  python3 plot_scripts/plot.py nouvelle_trajectoire --fgo all

Figures générées :
  - Figure 1 : GPS vs Odom (X, Y, Z) + RMSE + courbes FGO si demandées
  - Figure 2 : IMU accélérations vs vérité terrain
  - Figure 3 : IMU vitesses angulaires vs vérité terrain

Convention de nommage des CSV :
  csv_files/gps_<trajectoire>.csv
  csv_files/imu_<trajectoire>.csv
  csv_files/odom_<trajectoire>.csv
  csv_files/fgo_csv/fgo_gps_<trajectoire>.csv        (optionnel)
  csv_files/fgo_csv/fgo_gps_imu_<trajectoire>.csv    (optionnel)

Pour le cas "sol", les fichiers sont gps.csv / imu.csv / odom.csv (sans suffixe).
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
from plot_scripts.plot_sensors import fig1_gps_vs_odom, fig2_imu_accel, fig3_imu_gyro

_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(_ROOT, "csv_files")
FGO_DIR = os.path.join(CSV_DIR, "fgo_csv")


def build_paths(nom: str) -> dict:
    """
    Construit les chemins CSV pour une trajectoire donnée.
    Convention : 'sol' → gps.csv, sinon gps_<nom>.csv
    """
    suf = "" if nom == "sol" else f"_{nom}"
    return {
        "gps":      os.path.join(CSV_DIR, f"gps{suf}.csv"),
        "imu":      os.path.join(CSV_DIR, f"imu{suf}.csv"),
        "odom":     os.path.join(CSV_DIR, f"odom{suf}.csv"),
        "fgo_gps":  os.path.join(FGO_DIR, f"fgo_gps{suf}.csv"),
        "fgo_imu":  os.path.join(FGO_DIR, f"fgo_gps_imu{suf}.csv"),
        "statique": nom == "sol",
    }


def check_csv(paths: dict, nom: str) -> bool:
    """Vérifie que les 3 CSV capteurs existent."""
    manquants = [k for k in ["gps", "imu", "odom"]
                 if not os.path.isfile(paths[k])]
    if manquants:
        print(f"[Erreur] CSV manquants pour '{nom}' : "
              f"{[os.path.basename(paths[k]) for k in manquants]}")
        print(f"[Info]  Dossier scanné : {CSV_DIR}")
        return False
    return True


def main():
    # ── Parsing des arguments ─────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Plot universel des capteurs et FGO",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "trajectoire",
        help="Nom de la trajectoire (ex: sol, vertical, carre...)"
    )
    parser.add_argument(
        "--fgo",
        choices=["gps", "gps_imu", "all"],
        default=None,
        help=(
            "Ajoute les courbes FGO sur la figure GPS :\n"
            "  gps     → FGO GPS seul\n"
            "  gps_imu → FGO GPS+IMU\n"
            "  all     → les deux"
        )
    )
    args = parser.parse_args()
    nom  = args.trajectoire

    # ── Construction des chemins ──────────────────────────────────────────
    paths = build_paths(nom)

    # Vérifie que les CSV capteurs existent
    if not check_csv(paths, nom):
        sys.exit(1)

    # ── Détection des CSV FGO demandés ────────────────────────────────────
    fgo_gps = None
    fgo_imu = None

    if args.fgo in ("gps", "all"):
        if os.path.isfile(paths["fgo_gps"]):
            fgo_gps = paths["fgo_gps"]
            print(f"[Plot] FGO GPS détecté : {os.path.basename(fgo_gps)}")
        else:
            print(f"[Avertissement] FGO GPS non trouvé : "
                  f"{os.path.basename(paths['fgo_gps'])}")
            print(f"[Info] Lance d'abord : python3 fusion/fgo_gps.py {nom}")

    if args.fgo in ("gps_imu", "all"):
        if os.path.isfile(paths["fgo_imu"]):
            fgo_imu = paths["fgo_imu"]
            print(f"[Plot] FGO GPS+IMU détecté : {os.path.basename(fgo_imu)}")
        else:
            print(f"[Avertissement] FGO GPS+IMU non trouvé : "
                  f"{os.path.basename(paths['fgo_imu'])}")
            print(f"[Info] Lance d'abord : python3 fusion/fgo_gps_imu.py {nom}")

    # ── Génération des figures ────────────────────────────────────────────
    print(f"\n[Plot] Trajectoire : '{nom}' "
          f"({'statique' if paths['statique'] else 'en vol'})")

    # Figure 1 : GPS + FGO(s) vs Odom
    fig1_gps_vs_odom(
        paths["gps"], paths["odom"],
        titre=nom,
        fgo_csv=fgo_gps,
        fgo_imu_csv=fgo_imu
    )

    # Figure 2 : IMU accélérations
    fig2_imu_accel(
        paths["imu"], paths["odom"],
        statique=paths["statique"],
        titre=nom
    )

    # Figure 3 : IMU vitesses angulaires
    fig3_imu_gyro(
        paths["imu"], paths["odom"],
        statique=paths["statique"],
        titre=nom
    )

    plt.show()


if __name__ == "__main__":
    main()