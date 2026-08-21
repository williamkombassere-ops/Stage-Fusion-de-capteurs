#!/usr/bin/env python3
"""
fgo_gps.py
----------
Factor Graph Optimization (FGO) GPS seul en mode batch.
Lit les CSV de n'importe quelle trajectoire, optimise avec GTSAM
et sauvegarde le résultat dans csv_files/fgo_csv/.

Facteurs utilisés :
  - PriorFactor  : ancre le premier noeud à la position initiale
  - GPSFactor    : contrainte absolue GPS sur chaque noeud
  - BetweenFactor: modèle de mouvement (marche aléatoire) entre noeuds consécutifs

Usage :
  python3 fusion/fgo_gps.py                        ← détecte tous les groupes CSV
  python3 fusion/fgo_gps.py vertical               ← optimise uniquement 'vertical'
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import csv
import numpy as np
import pandas as pd
import gtsam
from gtsam import symbol_shorthand

_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(_ROOT, "csv_files")
FGO_DIR = os.path.join(CSV_DIR, "fgo_csv")

# Raccourci GTSAM pour les variables de position
X = symbol_shorthand.X

# ─── Paramètres modifiables ──────────────────────────────────────────────────
SIGMA_PRIOR  = 0.001   # m  — très confiant sur la position initiale
SIGMA_GPS    = 1.0    # m  — RMSE mesuré sur nos capteurs Gazebo
SIGMA_BETWEEN = 0.05   # plus petit que SIGMA_GPS pour lisser    # m  — modèle de mouvement entre noeuds (marche aléatoire)
DELTA_K      = 3       # sous-échantillonnage : 1 noeud tous les DELTA_K points GPS
# ─────────────────────────────────────────────────────────────────────────────


def gps_to_meters(lat, lon, alt):
    """
    Convertit lat/lon/alt en (X, Y, Z) en mètres depuis l'origine. Utilise la médiane des 10 premières mesures comme référence robuste.
    """
    R     = 6_371_000.0
    N_REF = min(10, len(lat))
    lat_rad = np.radians(lat)
    lat0  = np.median(lat_rad[:N_REF])
    lon0  = np.median(lon[:N_REF])
    alt0  = np.median(alt[:N_REF])
    dx = (lon - lon0) * np.cos(lat0) * np.radians(R)
    dy = (lat - np.median(lat[:N_REF])) * np.radians(R)
    dz = alt - alt0
    return dx, dy, dz


def detecter_trajectoires():
    """Détecte les groupes GPS/Odom disponibles dans csv_files/."""
    import re
    trajectoires = {}
    for f in os.listdir(CSV_DIR):
        m = re.match(r"^gps(?:_(.+))?\.csv$", f)
        if m:
            nom  = m.group(1) or "sol"
            suf  = f"_{nom}" if nom != "sol" else ""
            gps  = os.path.join(CSV_DIR, f"gps{suf}.csv")
            odom = os.path.join(CSV_DIR, f"odom{suf}.csv")
            if os.path.isfile(gps) and os.path.isfile(odom):
                trajectoires[nom] = {"gps": gps, "odom": odom}
    return trajectoires


def run_fgo_gps(nom: str, gps_csv: str, odom_csv: str):
    """
    Construit et optimise le graphe de facteurs GPS seul pour une trajectoire.
    Sauvegarde le résultat dans fgo_csv/fgo_gps_{nom}.csv.
    """
    print(f"\n[FGO GPS] Trajectoire : {nom}")

    # ── Chargement et conversion GPS → mètres ────────────────────────────
    df_gps = pd.read_csv(gps_csv).sort_values("t")
    dx, dy, dz = gps_to_meters(
        df_gps["lat"].to_numpy(),
        df_gps["lon"].to_numpy(),
        df_gps["alt"].to_numpy()
    )

    # Recalage GPS sur le repère Odom (l'Odom dit la vérité)
    df_odom_ref = pd.read_csv(odom_csv).sort_values("t").reset_index(drop=True)
    t_od_ref    = df_odom_ref["t"].to_numpy()
    t_gps_abs   = df_gps["t"].to_numpy()
    x_offset = float(np.interp(t_gps_abs[0], t_od_ref, df_odom_ref["x"].to_numpy()))
    y_offset = float(np.interp(t_gps_abs[0], t_od_ref, df_odom_ref["y"].to_numpy()))
    z_offset = float(np.interp(t_gps_abs[0], t_od_ref, df_odom_ref["z"].to_numpy()))
    dx += x_offset
    dy += y_offset
    dz += z_offset

    # Correction du biais GPS sur X et Y depuis les N_BIAS premières mesures
    N_BIAS = 5
    x_odom_init = np.array([float(np.interp(t_gps_abs[i], t_od_ref,
                            df_odom_ref["x"].to_numpy())) for i in range(N_BIAS)])
    y_odom_init = np.array([float(np.interp(t_gps_abs[i], t_od_ref,
                            df_odom_ref["y"].to_numpy())) for i in range(N_BIAS)])
    biais_x = np.mean(dx[:N_BIAS] - x_odom_init)
    biais_y = np.mean(dy[:N_BIAS] - y_odom_init)
    dx -= biais_x
    dy -= biais_y
    print(f"[FGO GPS] Biais GPS estimé : X={biais_x:.4f}m  Y={biais_y:.4f}m")
    t_gps = (df_gps["t"] - df_gps["t"].iloc[0]).to_numpy()

    # Sous-échantillonnage : 1 noeud tous les DELTA_K points GPS
    indices = list(range(0, len(dx), DELTA_K))
    N       = len(indices)
    print(f"[FGO GPS] {len(dx)} mesures GPS → {N} noeuds (DELTA_K={DELTA_K})")

    # ── Construction du graphe de facteurs ────────────────────────────────
    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # Matrices de covariance (diagonales)
    cov_prior   = gtsam.noiseModel.Diagonal.Sigmas(
                      np.array([SIGMA_PRIOR]*3))
    cov_gps     = gtsam.noiseModel.Diagonal.Sigmas(
                      np.array([SIGMA_GPS]*3))
    cov_between = gtsam.noiseModel.Diagonal.Sigmas(
                      np.array([SIGMA_BETWEEN]*3))

    for i, idx in enumerate(indices):
        pos = gtsam.Point3(dx[idx], dy[idx], dz[idx])

        # Valeur initiale : position GPS brute
        # Initialisation à zéro pour forcer l'optimisation
        values.insert(X(i), gtsam.Point3(0.0, 0.0, 0.0))

        # ── Facteur Prior sur le premier noeud ───────────────────────────
        if i == 0:
            graph.add(gtsam.PriorFactorPoint3(X(0), pos, cov_prior))

        # ── Facteur GPS : contrainte absolue ─────────────────────────────
        graph.add(gtsam.PriorFactorPoint3(X(i), pos, cov_gps))

        # ── Facteur Between : continuité entre noeuds consécutifs ────────
        if i > 0:
            prev_idx = indices[i - 1]
            deplacement = gtsam.Point3(
                dx[idx] - dx[prev_idx],
                dy[idx] - dy[prev_idx],
                dz[idx] - dz[prev_idx]
            )
            graph.add(gtsam.BetweenFactorPoint3(
                X(i-1), X(i), deplacement, cov_between))

    print(f"[FGO GPS] Graphe : {graph.size()} facteurs, {N} noeuds")

    # ── Optimisation Levenberg-Marquardt ──────────────────────────────────
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(500)
    params.setVerbosity("SILENT")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    result    = optimizer.optimize()

    print(f"[FGO GPS] Erreur initiale : {graph.error(values):.4f}")
    print(f"[FGO GPS] Erreur finale   : {graph.error(result):.4f}")

    # ── Calcul RMSE vs vérité terrain (Odom) ─────────────────────────────
    df_odom = pd.read_csv(odom_csv).sort_values("t")
    t_odom  = (df_odom["t"] - df_odom["t"].iloc[0]).to_numpy()
    t_fgo   = t_gps[indices]

    x_odom = np.interp(t_fgo, t_odom, df_odom["x"].to_numpy())
    y_odom = np.interp(t_fgo, t_odom, df_odom["y"].to_numpy())
    z_odom = np.interp(t_fgo, t_odom, df_odom["z"].to_numpy())

    # RMSE GPS brut vs Odom
    rmse_gps = np.sqrt(np.mean(
        (dx[indices] - x_odom)**2 +
        (dy[indices] - y_odom)**2 +
        (dz[indices] - z_odom)**2
    ))

    # Extraction résultats FGO
    x_fgo = np.array([result.atPoint3(X(i))[0] for i in range(N)])
    y_fgo = np.array([result.atPoint3(X(i))[1] for i in range(N)])
    z_fgo = np.array([result.atPoint3(X(i))[2] for i in range(N)])

    # RMSE FGO vs Odom
    rmse_fgo = np.sqrt(np.mean(
        (x_fgo - x_odom)**2 +
        (y_fgo - y_odom)**2 +
        (z_fgo - z_odom)**2
    ))

    gain = (rmse_gps - rmse_fgo) / rmse_gps * 100
    print(f"[FGO GPS] RMSE GPS brut : {rmse_gps:.4f} m")
    print(f"[FGO GPS] RMSE FGO      : {rmse_fgo:.4f} m")
    print(f"[FGO GPS] Gain          : {gain:.1f} %")

    # ── Sauvegarde CSV ────────────────────────────────────────────────────
    os.makedirs(FGO_DIR, exist_ok=True)
    out_csv = os.path.join(FGO_DIR, f"fgo_gps_{nom}.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "z"])
        writer.writeheader()
        for i in range(N):
            writer.writerow({
                "t": round(t_fgo[i], 4),
                "x": round(x_fgo[i], 6),
                "y": round(y_fgo[i], 6),
                "z": round(z_fgo[i], 6),
            })

    print(f"[FGO GPS] Sauvegardé : {out_csv} ({N} lignes)")
    return out_csv


def main():
    """
    Détecte toutes les trajectoires disponibles et optimise chacune. Si un argument est passé, optimise uniquement cette trajectoire.
    """
    trajectoires = detecter_trajectoires()

    if not trajectoires:
        print("[Erreur] Aucun groupe GPS/Odom trouvé dans csv_files/")
        return

    # Filtre si un nom est passé en argument
    if len(sys.argv) > 1:
        nom = sys.argv[1]
        if nom not in trajectoires:
            print(f"[Erreur] Trajectoire '{nom}' introuvable. "
                  f"Disponibles : {list(trajectoires.keys())}")
            return
        trajectoires = {nom: trajectoires[nom]}

    for nom, fichiers in trajectoires.items():
        run_fgo_gps(nom, fichiers["gps"], fichiers["odom"])


if __name__ == "__main__":
    main()