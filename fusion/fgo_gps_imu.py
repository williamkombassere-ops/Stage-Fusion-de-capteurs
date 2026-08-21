#!/usr/bin/env python3
"""
fgo_gps_imu.py
--------------
Factor Graph Optimization (FGO) GPS + IMU en mode batch.
Utilise la pré-intégration IMU de GTSAM pour créer des BetweenFactors
cinématiquement cohérents entre les noeuds GPS.

Variables d'état par noeud :
  - Pose  (position + orientation) via gtsam.Pose3
  - Vitesse (vx, vy, vz)
  - Biais IMU (accéléromètre + gyroscope)

Facteurs :
  - PriorFactor     : ancre le premier noeud (position, vitesse, biais)
  - GPSFactor       : contrainte absolue de position GPS
  - ImuFactor       : pré-intégration IMU entre deux noeuds consécutifs

Usage :
  python3 fusion/fgo_gps_imu.py              ← toutes les trajectoires
  python3 fusion/fgo_gps_imu.py vertical     ← uniquement 'vertical'
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

# Raccourcis GTSAM
X = symbol_shorthand.X   # Pose3  (position + orientation)
V = symbol_shorthand.V   # vitesse
B = symbol_shorthand.B   # biais IMU

# ─── Paramètres modifiables ──────────────────────────────────────────────────
SIGMA_PRIOR_POS  = 0.15    # m  — même ordre que SIGMA_GPS_XY (biais GPS initial)
SIGMA_PRIOR_VEL  = 0.01    # m/s     — vitesse initiale ≈ nulle
SIGMA_PRIOR_BIAS = 0.01    # m/s²    — biais initiaux ≈ nuls

# Covariance GPS par axe — calibrée depuis les RMSE mesurés sur nos capteurs Gazebo
SIGMA_GPS_XY     = 0.12    # m  — RMSE mesuré : X=0.09m, Y=0.12m (drone au sol)
SIGMA_GPS_Z      = 0.35    # m  — RMSE mesuré : Z=0.34m (vol vertical, GPS lent à 2.5Hz)

# Bruit IMU (cohérents avec le SDF Gazebo)
SIGMA_ACC_NOISE  = 0.05    # m/s²    — bruit blanc accéléromètre
SIGMA_GYR_NOISE  = 0.02    # rad/s   — bruit blanc gyroscope
SIGMA_ACC_BIAS   = 0.001   # m/s²/√s — marche aléatoire biais accéléro
SIGMA_GYR_BIAS   = 0.001   # rad/s/√s— marche aléatoire biais gyro

GRAVITY          = 9.8     # m/s²    — configuré dans le SDF
DELTA_K          = 3       # sous-échantillonnage GPS (1 noeud / DELTA_K mesures)
# ─────────────────────────────────────────────────────────────────────────────


def gps_to_meters(lat, lon, alt):
    """
    Convertit lat/lon/alt en (X, Y, Z) en mètres depuis l'origine.
    Utilise la médiane des 10 premières mesures comme référence robuste.
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
    """Détecte les groupes GPS/IMU/Odom disponibles dans csv_files/."""
    import re
    trajectoires = {}
    for f in os.listdir(CSV_DIR):
        m = re.match(r"^gps(?:_(.+))?\.csv$", f)
        if m:
            nom = m.group(1) or "sol"
            suf = f"_{nom}" if nom != "sol" else ""
            gps  = os.path.join(CSV_DIR, f"gps{suf}.csv")
            imu  = os.path.join(CSV_DIR, f"imu{suf}.csv")
            odom = os.path.join(CSV_DIR, f"odom{suf}.csv")
            if all(os.path.isfile(f) for f in [gps, imu, odom]):
                trajectoires[nom] = {"gps": gps, "imu": imu, "odom": odom}
    return trajectoires


def run_fgo_gps_imu(nom: str, gps_csv: str, imu_csv: str, odom_csv: str):
    """
    Construit et optimise le graphe GPS + IMU pour une trajectoire.
    Sauvegarde dans fgo_csv/fgo_gps_imu_{nom}.csv.
    """
    print(f"\n[FGO GPS+IMU] Trajectoire : {nom}")

    # ── Chargement des données ────────────────────────────────────────────
    df_gps = pd.read_csv(gps_csv).sort_values("t").reset_index(drop=True)
    df_imu = pd.read_csv(imu_csv).sort_values("t").reset_index(drop=True)

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

    # Estimation et correction du biais GPS sur X et Y
    # On utilise les N_BIAS premières mesures GPS (drone encore au sol)
    # où la vraie position est connue depuis l'Odom
    N_BIAS = 5
    x_odom_init = np.array([float(np.interp(t_gps_abs[i], t_od_ref,
                            df_odom_ref["x"].to_numpy())) for i in range(N_BIAS)])
    y_odom_init = np.array([float(np.interp(t_gps_abs[i], t_od_ref,
                            df_odom_ref["y"].to_numpy())) for i in range(N_BIAS)])
    biais_x = np.mean(dx[:N_BIAS] - x_odom_init)
    biais_y = np.mean(dy[:N_BIAS] - y_odom_init)
    dx -= biais_x
    dy -= biais_y
    print(f"[FGO GPS+IMU] Biais GPS estimé : X={biais_x:.4f}m  Y={biais_y:.4f}m")
    t_gps = df_gps["t"].to_numpy()
    t_imu = df_imu["t"].to_numpy()

    # Sous-échantillonnage GPS
    indices = list(range(0, len(dx), DELTA_K))
    N       = len(indices)
    print(f"[FGO GPS+IMU] {len(dx)} GPS → {N} noeuds | {len(t_imu)} IMU")

    # ── Paramètres de pré-intégration IMU ────────────────────────────────
    imu_params = gtsam.PreintegrationParams.MakeSharedU(GRAVITY)

    # Matrices de bruit IMU
    imu_params.setAccelerometerCovariance(
        np.eye(3) * SIGMA_ACC_NOISE**2)
    imu_params.setGyroscopeCovariance(
        np.eye(3) * SIGMA_GYR_NOISE**2)
    imu_params.setIntegrationCovariance(
        np.eye(3) * 1e-8)  # erreur d'intégration numérique

    # Biais initiaux (supposés nuls au départ)
    prior_bias = gtsam.imuBias.ConstantBias(
        np.zeros(3), np.zeros(3))

    # ── Construction du graphe de facteurs ────────────────────────────────
    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # Matrices de covariance
    cov_prior_pose = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e-4, 1e-4, 1e-4,              # rotation (rad)
                  SIGMA_PRIOR_POS]*3)[:6])        # position (m)
    cov_prior_vel  = gtsam.noiseModel.Isotropic.Sigma(3, SIGMA_PRIOR_VEL)
    cov_prior_bias = gtsam.noiseModel.Isotropic.Sigma(6, SIGMA_PRIOR_BIAS)
    # Covariance GPS diagonale : moins confiant sur Z (GPS lent vs montée rapide)
    cov_gps = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([SIGMA_GPS_XY, SIGMA_GPS_XY, SIGMA_GPS_Z]))

    # ── Calcul des vitesses initiales depuis l'Odom (dérivée numérique) ────
    # Charge l'Odom pour initialiser les vitesses — évite la vitesse nulle
    df_odom_init = pd.read_csv(odom_csv).sort_values("t").reset_index(drop=True)
    t_od  = df_odom_init["t"].to_numpy()
    vx_od = np.gradient(df_odom_init["x"].to_numpy(), t_od)
    vy_od = np.gradient(df_odom_init["y"].to_numpy(), t_od)
    vz_od = np.gradient(df_odom_init["z"].to_numpy(), t_od)

    def get_vel_at(t_query):
        """Interpolation de la vitesse Odom à l'instant t_query."""
        return np.array([
            float(np.interp(t_query, t_od, vx_od)),
            float(np.interp(t_query, t_od, vy_od)),
            float(np.interp(t_query, t_od, vz_od)),
        ])

    # ── Facteur Prior sur le noeud 0 ─────────────────────────────────────
    pose0 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(dx[0], dy[0], dz[0]))
    vel0  = get_vel_at(t_gps[0])

    graph.add(gtsam.PriorFactorPose3(X(0), pose0, cov_prior_pose))
    # Prior vitesse : on fait confiance à la dérivée Odom (sigma plus grand)
    cov_prior_vel = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    graph.add(gtsam.PriorFactorVector(V(0), vel0,  cov_prior_vel))
    graph.add(gtsam.PriorFactorConstantBias(B(0), prior_bias, cov_prior_bias))

    values.insert(X(0), pose0)
    values.insert(V(0), vel0)
    values.insert(B(0), prior_bias)

    # ── Facteur GPS noeud 0 ───────────────────────────────────────────────
    graph.add(gtsam.GPSFactor(
        X(0),
        gtsam.Point3(dx[indices[0]], dy[indices[0]], dz[indices[0]]),
        cov_gps))

    # ── Boucle sur les noeuds ─────────────────────────────────────────────
    for i in range(1, N):
        idx_prev = indices[i - 1]
        idx_curr = indices[i]

        t_start = t_gps[idx_prev]
        t_end   = t_gps[idx_curr]

        # Pré-intégration IMU entre t_start et t_end
        pim = gtsam.PreintegratedImuMeasurements(imu_params, prior_bias)

        # Sélectionne les mesures IMU dans la fenêtre [t_start, t_end]
        mask = (t_imu >= t_start) & (t_imu < t_end)
        imu_window = df_imu[mask]

        if len(imu_window) == 0:
            # Pas de mesure IMU → intègre un pas fictif nul
            dt = t_end - t_start
            pim.integrateMeasurement(np.zeros(3), np.zeros(3), dt)
        else:
            t_prev_imu = t_start
            for _, row in imu_window.iterrows():
                dt = row["t"] - t_prev_imu
                if dt > 0:
                    acc = np.array([row["ax"], row["ay"], row["az"]])  # GTSAM soustrait g en interne
                    gyr = np.array([row["wx"], row["wy"], row["wz"]])
                    pim.integrateMeasurement(acc, gyr, dt)
                t_prev_imu = row["t"]

        # ── Facteur IMU entre noeud i-1 et noeud i ───────────────────────
        graph.add(gtsam.ImuFactor(
            X(i-1), V(i-1),
            X(i),   V(i),
            B(i-1),
            pim))

        # ── Facteur biais : évolution lente (marche aléatoire) ───────────
        cov_bias_between = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([SIGMA_ACC_BIAS] * 3 + [SIGMA_GYR_BIAS] * 3) *
            np.sqrt(t_end - t_start))
        graph.add(gtsam.BetweenFactorConstantBias(
            B(i-1), B(i),
            gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
            cov_bias_between))

        # ── Facteur GPS ───────────────────────────────────────────────────
        graph.add(gtsam.GPSFactor(
            X(i),
            gtsam.Point3(dx[idx_curr], dy[idx_curr], dz[idx_curr]),
            cov_gps))

        # ── Valeurs initiales pour ce noeud (vitesse depuis Odom) ──────────
        pose_i = gtsam.Pose3(
            gtsam.Rot3(),
            gtsam.Point3(dx[idx_curr], dy[idx_curr], dz[idx_curr]))
        vel_i  = get_vel_at(t_gps[idx_curr])   # vitesse Odom interpolée
        values.insert(X(i), pose_i)
        values.insert(V(i), vel_i)
        values.insert(B(i), prior_bias)

    print(f"[FGO GPS+IMU] Graphe : {graph.size()} facteurs, {N} noeuds")

    # ── Optimisation Levenberg-Marquardt ──────────────────────────────────
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(500)
    params.setVerbosity("SILENT")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    result    = optimizer.optimize()

    print(f"[FGO GPS+IMU] Erreur initiale : {graph.error(values):.4f}")
    print(f"[FGO GPS+IMU] Erreur finale   : {graph.error(result):.4f}")

    # ── Extraction des résultats ──────────────────────────────────────────
    t_fgo = t_gps[indices] - t_gps[0]   # temps relatif
    x_fgo = np.array([result.atPose3(X(i)).translation()[0] for i in range(N)])
    y_fgo = np.array([result.atPose3(X(i)).translation()[1] for i in range(N)])
    z_fgo = np.array([result.atPose3(X(i)).translation()[2] for i in range(N)])

    # ── Calcul RMSE vs vérité terrain (Odom) ─────────────────────────────
    df_odom = pd.read_csv(odom_csv).sort_values("t")
    t_odom  = (df_odom["t"] - df_odom["t"].iloc[0]).to_numpy()

    x_od = np.interp(t_fgo, t_odom, df_odom["x"].to_numpy())
    y_od = np.interp(t_fgo, t_odom, df_odom["y"].to_numpy())
    z_od = np.interp(t_fgo, t_odom, df_odom["z"].to_numpy())

    # GPS brut RMSE
    dx_rel, dy_rel, dz_rel = dx[indices], dy[indices], dz[indices]
    rmse_gps = np.sqrt(np.mean((dx_rel-x_od)**2 + (dy_rel-y_od)**2 + (dz_rel-z_od)**2))

    # FGO RMSE
    rmse_fgo = np.sqrt(np.mean((x_fgo-x_od)**2 + (y_fgo-y_od)**2 + (z_fgo-z_od)**2))

    gain = (rmse_gps - rmse_fgo) / rmse_gps * 100
    print(f"[FGO GPS+IMU] RMSE GPS brut : {rmse_gps:.4f} m")
    print(f"[FGO GPS+IMU] RMSE FGO      : {rmse_fgo:.4f} m")
    print(f"[FGO GPS+IMU] Gain          : {gain:.1f} %")

    # ── Sauvegarde CSV ────────────────────────────────────────────────────
    os.makedirs(FGO_DIR, exist_ok=True)
    out_csv = os.path.join(FGO_DIR, f"fgo_gps_imu_{nom}.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "z"])
        writer.writeheader()
        for i in range(N):
            writer.writerow({
                "t": round(float(t_fgo[i]), 4),
                "x": round(float(x_fgo[i]), 6),
                "y": round(float(y_fgo[i]), 6),
                "z": round(float(z_fgo[i]), 6),
            })

    print(f"[FGO GPS+IMU] Sauvegardé : {out_csv} ({N} lignes)")
    return out_csv


def main():
    """Détecte toutes les trajectoires et optimise chacune."""
    trajectoires = detecter_trajectoires()

    if not trajectoires:
        print("[Erreur] Aucun groupe GPS/IMU/Odom trouvé dans csv_files/")
        return

    if len(sys.argv) > 1:
        nom = sys.argv[1]
        if nom not in trajectoires:
            print(f"[Erreur] Trajectoire '{nom}' introuvable. "
                  f"Disponibles : {list(trajectoires.keys())}")
            return
        trajectoires = {nom: trajectoires[nom]}

    for nom, f in trajectoires.items():
        try:
            run_fgo_gps_imu(nom, f["gps"], f["imu"], f["odom"])
        except Exception as e:
            import traceback
            print(f"[ERREUR] Trajectoire '{nom}' : {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()