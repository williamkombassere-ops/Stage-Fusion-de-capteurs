#!/usr/bin/env python3
"""
plot_euler.py
-------------
Convertit le quaternion Odom en angles d'Euler (roll, pitch, yaw) et les
trace en fonction du temps, à côté de la position (x, y, z) — 6 tracés
au total, alignés sur le même axe temps. Objectif : voir directement si
un angle (roll/pitch/yaw) correspond à un vrai déplacement ou non, pour
comprendre le comportement réel du drone d'un coup d'œil.

Formules de conversion quaternion → Euler (convention ZYX) :
  roll  (φ) = arctan2(2(qw*qx + qy*qz), 1 - 2(qx² + qy²))
  pitch (θ) = arcsin(2(qw*qy - qz*qx))
  yaw   (ψ) = arctan2(2(qw*qz + qx*qy), 1 - 2(qy² + qz²))

Usage :
  python3 plot_scripts/plot_euler.py <trajectoire>

Exemples :
  python3 plot_scripts/plot_euler.py rotation   ← vol avec rotation yaw
  python3 plot_scripts/plot_euler.py vertical
  python3 plot_scripts/plot_euler.py sol
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(_ROOT, "csv_files")
IMG_DIR = os.path.join(_ROOT, "images")


def quat_to_euler(qx, qy, qz, qw):
    """
    Convertit des tableaux de quaternions en angles d'Euler (rad)
    selon la convention ZYX (yaw-pitch-roll).

    Formules :
      roll  = arctan2(2(qw*qx + qy*qz), 1 - 2(qx² + qy²))
      pitch = arcsin(clip(2(qw*qy - qz*qx), -1, 1))
      yaw   = arctan2(2(qw*qz + qx*qy), 1 - 2(qy² + qz²))
    """
    # Roll (φ) — rotation autour de l'axe X
    roll  = np.arctan2(2*(qw*qx + qy*qz),
                       1 - 2*(qx**2 + qy**2))

    # Pitch (θ) — rotation autour de l'axe Y
    # clip pour éviter les erreurs numériques hors [-1, 1]
    pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))

    # Yaw (ψ) — rotation autour de l'axe Z
    yaw   = np.arctan2(2*(qw*qz + qx*qy),
                       1 - 2*(qy**2 + qz**2))

    return roll, pitch, yaw


def plot_euler(trajectoire: str):
    """
    Charge le CSV Odom de la trajectoire donnée, calcule les angles
    d'Euler, et trace 6 subplots (roll, pitch, yaw, x, y, z) partageant
    le même axe temps.
    Sauvegarde dans images/euler_<trajectoire>.pdf
    """
    # ── Chargement CSV ────────────────────────────────────────────────────
    suf     = "" if trajectoire == "sol" else f"_{trajectoire}"
    odom_csv = os.path.join(CSV_DIR, f"odom{suf}.csv")

    if not os.path.isfile(odom_csv):
        print(f"[Erreur] Fichier introuvable : {odom_csv}")
        return

    df = pd.read_csv(odom_csv).sort_values("t")

    # Vérification des colonnes nécessaires (quaternion + position)
    for col in ["qx", "qy", "qz", "qw", "x", "y", "z"]:
        if col not in df.columns:
            print(f"[Erreur] Colonne '{col}' absente — refais la collecte.")
            return

    # ── Conversion quaternion → Euler ─────────────────────────────────────
    roll, pitch, yaw = quat_to_euler(
        df["qx"].to_numpy(),
        df["qy"].to_numpy(),
        df["qz"].to_numpy(),
        df["qw"].to_numpy()
    )

    # Conversion en degrés pour la lisibilité
    roll_deg  = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg   = np.degrees(yaw)

    # Position brute (Odom, vérité terrain)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()

    # Temps relatif (part de 0)
    t = (df["t"] - df["t"].iloc[0]).to_numpy()

    # ── Figure : 6 subplots (3 angles + 3 positions), même axe temps ──────
    fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f"Angles d'Euler + Position — {trajectoire}\n"
                 r"(convention ZYX : yaw $\to$ pitch $\to$ roll)",
                 fontsize=13, fontweight="bold")

    configs = [
        (roll_deg,  "tab:blue",   r"Roll $\phi$ (°)",   "Roll"),
        (pitch_deg, "tab:orange", r"Pitch $\theta$ (°)", "Pitch"),
        (yaw_deg,   "tab:green",  r"Yaw $\psi$ (°)",    "Yaw"),
        (x,         "tab:red",    "X (m)",              "X"),
        (y,         "tab:purple", "Y (m)",               "Y"),
        (z,         "tab:brown",  "Z (m)",               "Z"),
    ]

    for ax, (data, color, ylabel, label) in zip(axes, configs):
        ax.plot(t, data, color=color, linewidth=1.2, label=label)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=9)

        # Annotation de la valeur max (utile pour repérer les pics)
        idx_max = np.argmax(np.abs(data))
        if np.abs(data[idx_max]) > 1e-9:  # évite une annotation illisible sur du quasi-zéro
            ax.annotate(f"{data[idx_max]:.3g}",
                        xy=(t[idx_max], data[idx_max]),
                        xytext=(t[idx_max] + 0.5, data[idx_max] * 1.05),
                        fontsize=8, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    axes[-1].set_xlabel("Temps (s)")
    plt.tight_layout()

    # ── Sauvegarde ────────────────────────────────────────────────────────
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, f"euler_{trajectoire}.pdf")
    plt.savefig(out)
    print(f"[Plot] Angles d'Euler + Position sauvegardés : {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot des angles d'Euler + position depuis l'Odom")
    parser.add_argument(
        "trajectoire",
        help="Nom de la trajectoire (ex: rotation, vertical, sol)")
    args = parser.parse_args()

    plot_euler(args.trajectoire)