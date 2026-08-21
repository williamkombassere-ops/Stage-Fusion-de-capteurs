#!/usr/bin/env python3
"""
fgo_imu.py
-----------
Estime la position du drone à partir de l'IMU SEULE (aucun GPS, aucun
FGO) par double intégration numérique de l'accélération — la méthode
dite de "navigation inertielle pure" (dead reckoning).

Principe :
  1. On suppose l'orientation quasi constante (drone proche du niveau,
     valable pour les vols vertical/carré — pas pour un vol avec forte
     rotation) : le repère capteur est approximé au repère monde.
  2. On soustrait la gravité (9.8 m/s² sur Z) de l'accélération mesurée.
  3. On intègre une fois : accélération → vitesse (cumul trapézoïdal).
  4. On intègre une seconde fois : vitesse → position.
  Vitesse et position initiales sont prises nulles (drone au repos au
  décollage).

Contrairement au GPS, cette estimation n'est jamais recalée par une
mesure absolue : chaque petite erreur (bruit, biais) s'accumule sans
correction — la position dérive avec le temps. C'est exactement ce
que ce script doit mettre en évidence, avant l'introduction du FGO
GPS+IMU qui corrige cette dérive grâce aux mesures GPS absolues.

Usage :
  python3 fusion/fgo_imu.py vertical
  python3 fusion/fgo_imu.py               ← détecte toutes les trajectoires
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_scripts.plot_sensors import _plot_axe  # réutilise le tracé mesuré+vérité+erreur

_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(_ROOT, "csv_files")
IMU_DIR = os.path.join(CSV_DIR, "imu_seul_csv")
IMG_DIR = os.path.join(_ROOT, "images")

GRAVITY = 9.8


def integrer_imu(imu_csv: str) -> dict:
    """
    Double intégration de l'accélération IMU pour estimer la position.
    Retourne {"t": ..., "x": ..., "y": ..., "z": ...} (mêmes timestamps
    que le CSV IMU, temps relatif au premier échantillon).
    """
    df = pd.read_csv(imu_csv).sort_values("t")
    t  = (df["t"] - df["t"].iloc[0]).to_numpy()

    ax = df["ax"].to_numpy()
    ay = df["ay"].to_numpy()
    az = df["az"].to_numpy() - GRAVITY  # on retire la gravité (mesurée au repos)

    # ── Intégration 1 : accélération → vitesse (cumul trapézoïdal) ────────
    vx = np.concatenate(([0], np.cumsum(0.5 * (ax[1:] + ax[:-1]) * np.diff(t))))
    vy = np.concatenate(([0], np.cumsum(0.5 * (ay[1:] + ay[:-1]) * np.diff(t))))
    vz = np.concatenate(([0], np.cumsum(0.5 * (az[1:] + az[:-1]) * np.diff(t))))

    # ── Intégration 2 : vitesse → position (cumul trapézoïdal) ────────────
    x = np.concatenate(([0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * np.diff(t))))
    y = np.concatenate(([0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * np.diff(t))))
    z = np.concatenate(([0], np.cumsum(0.5 * (vz[1:] + vz[:-1]) * np.diff(t))))

    return {"t": t, "x": x, "y": y, "z": z}


def run_imu_seul(nom: str, imu_csv: str, odom_csv: str):
    """
    Calcule la position par IMU seul, trace le résultat vs Odom
    (position + erreur dans le même graphique par axe, comme pour le
    GPS), et sauvegarde le résultat dans un CSV.
    """
    print(f"\n[IMU seul] Trajectoire : {nom}")
    est = integrer_imu(imu_csv)

    df_odom = pd.read_csv(odom_csv).sort_values("t")
    t_odom  = (df_odom["t"] - df_odom["t"].iloc[0]).to_numpy()
    x_odom  = np.interp(est["t"], t_odom, df_odom["x"].to_numpy())
    y_odom  = np.interp(est["t"], t_odom, df_odom["y"].to_numpy())
    z_odom  = np.interp(est["t"], t_odom, df_odom["z"].to_numpy())

    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
    print(f"[IMU seul] RMSE X = {rmse(est['x'], x_odom):.4f} m")
    print(f"[IMU seul] RMSE Y = {rmse(est['y'], y_odom):.4f} m")
    print(f"[IMU seul] RMSE Z = {rmse(est['z'], z_odom):.4f} m")

    # ── Tracé : 3 sous-graphes (X, Y, Z), position IMU + vérité + erreur ──
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Position IMU seul (dead reckoning) vs Odom — {nom}",
                 fontsize=11, fontweight="bold")

    donnees = [
        ("X — Est/Ouest", "tab:blue",   est["x"], x_odom),
        ("Y — Nord/Sud",  "tab:orange", est["y"], y_odom),
        ("Z — Altitude",  "tab:green",  est["z"], z_odom),
    ]
    for ax, (label, color, mesure, verite) in zip(axes, donnees):
        _plot_axe(ax, est["t"], mesure, verite, label, "m", color)

    axes[-1].set_xlabel("Temps (s)")
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out_img = os.path.join(IMG_DIR, f"imu_seul_{nom}.pdf")
    plt.savefig(out_img)
    print(f"[IMU seul] Figure sauvegardée : {out_img}")

    # ── Sauvegarde CSV ──────────────────────────────────────────────────
    os.makedirs(IMU_DIR, exist_ok=True)
    out_csv = os.path.join(IMU_DIR, f"imu_seul_{nom}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "z"])
        writer.writeheader()
        for i in range(len(est["t"])):
            writer.writerow({
                "t": round(float(est["t"][i]), 4),
                "x": round(float(est["x"][i]), 6),
                "y": round(float(est["y"][i]), 6),
                "z": round(float(est["z"][i]), 6),
            })
    print(f"[IMU seul] CSV sauvegardé : {out_csv}")


def detecter_trajectoires():
    """Détecte les groupes IMU/Odom disponibles dans csv_files/."""
    import re
    trajectoires = {}
    for f in os.listdir(CSV_DIR):
        m = re.match(r"^gps(?:_(.+))?\.csv$", f)
        if m:
            nom  = m.group(1) or "sol"
            suf  = f"_{nom}" if nom != "sol" else ""
            imu  = os.path.join(CSV_DIR, f"imu{suf}.csv")
            odom = os.path.join(CSV_DIR, f"odom{suf}.csv")
            if os.path.isfile(imu) and os.path.isfile(odom):
                trajectoires[nom] = {"imu": imu, "odom": odom}
    return trajectoires


def main():
    trajectoires = detecter_trajectoires()
    if not trajectoires:
        print("[Erreur] Aucun groupe IMU/Odom trouvé dans csv_files/")
        return

    if len(sys.argv) > 1:
        nom = sys.argv[1]
        if nom not in trajectoires:
            print(f"[Erreur] Trajectoire '{nom}' introuvable. "
                  f"Disponibles : {list(trajectoires.keys())}")
            return
        trajectoires = {nom: trajectoires[nom]}

    for nom, f in trajectoires.items():
        run_imu_seul(nom, f["imu"], f["odom"])

    plt.show()


if __name__ == "__main__":
    main()