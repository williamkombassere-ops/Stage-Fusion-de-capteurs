#!/usr/bin/env python3
"""
fgo_real_flight.py
--------------------
FGO couple GPS+IMU, adapte au VOL REEL (donnees SD card Crazyflie +
mocap QTM), reprenant exactement la meme architecture de graphe que
fgo_IMU_GPS_coupled.py (Prior, ImuFactor, GPSFactor, biais random-walk).

DIFFERENCE avec fgo_IMU_GPS_coupled.py (simulation) :
  - Le "GPS" simule ici (gps_reel.csv) est directement en METRES
    locaux (mocap + bruit gaussien synthetique, voir mocap_to_csv.py)
    -- PAS en lat/lon/alt. On saute donc entierement la fonction
    gps_to_meters() : plus sur, car cette fonction suppose une
    latitude de reference (celle du monde Gazebo simule) qu'on ne
    connait pas pour un vrai vol -- l'utiliser telle quelle aurait pu
    introduire une erreur d'echelle silencieuse.
  - L'IMU reelle (imu_reel.csv) est deja en unites SI (m/s^2, rad/s)
    -- conversion faite par bin_to_imu_csv.py.
  - Les CSV gps_reel/imu_reel/odom_reel doivent etre sur la MEME
    horloge (Crazyflie, secondes) -- voir les 2 scripts de conversion.

Usage :
  python3 vol_reel/fgo_real_flight.py
  (lit automatiquement csv_files/gps_reel.csv, imu_reel.csv, odom_reel.csv)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gtsam
from gtsam import symbol_shorthand
from gtsam import PreintegrationParams, PreintegratedImuMeasurements
from gtsam import Pose3, Rot3, Point3
from gtsam.imuBias import ConstantBias

X = symbol_shorthand.X
V = symbol_shorthand.V
B = symbol_shorthand.B

_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(_ROOT, "csv_files")
OUT_DIR = os.path.join(CSV_DIR, "fgo_real_flight_csv")
IMG_DIR = os.path.join(_ROOT, "images")

GRAVITY = 9.8

# ─── Parametres -- IDENTIQUES a fgo_IMU_GPS_coupled.py (config qui a donne
# les meilleurs resultats sur le triangle simule) ──────────────────────────
SIGMA_GPS        = 1.0
N_BIAS           = 15
SIGMA_PRIOR_POS  = SIGMA_GPS / np.sqrt(N_BIAS)
SIGMA_PRIOR_ROT  = 0.1
SIGMA_PRIOR_VEL  = 0.1
SIGMA_PRIOR_BIAS = 0.1

ACCEL_SIGMA      = 0.173
GYRO_SIGMA       = 0.02
INTEGRATION_COV  = 1e-8
BIAS_RW_SIGMA    = 5e-3

# ── INFLATION DU BRUIT IMU (specifique aux donnees REELLES) ────────────────
# ACCEL_SIGMA/GYRO_SIGMA ci-dessus ont ete calibres pour le bruit SIMULE
# (Gazebo, IMU tactique propre). Un vrai IMU Crazyflie embarque en plus les
# VIBRATIONS des moteurs/helices -- un bruit haute frequence bien plus grand
# et de nature differente (pas un bruit blanc gaussien propre). Sans
# inflation, le graphe fait beaucoup trop confiance a l'IMU reel par rapport
# au GPS, ce qui produit des boucles/derives (meme symptome deja diagnostique
# sur le carre simule, ici amplifie car le bruit reel est structurellement
# plus grand que la valeur theorique).
NOISE_INFLATION_ACCEL = 1.0
NOISE_INFLATION_GYRO  = 1.0

DELTA_K          = 1
# ────────────────────────────────────────────────────────────────────────


def run_fgo_real(gps_csv, imu_csv, odom_csv, nom="reel"):
    print(f"\n[FGO REEL] Vol : {nom}")

    # ── Chargement GPS (deja en metres locaux, PAS de conversion lat/lon) ──
    df_gps = pd.read_csv(gps_csv).sort_values("t").reset_index(drop=True)
    dx = df_gps["x"].to_numpy(dtype=float)
    dy = df_gps["y"].to_numpy(dtype=float)
    dz = df_gps["z"].to_numpy(dtype=float)
    t_gps_abs = df_gps["t"].to_numpy()

    df_odom = pd.read_csv(odom_csv).sort_values("t").reset_index(drop=True)
    t_od_ref = df_odom["t"].to_numpy()

    # ── Verification de recouvrement temporel (sanity check important pour
    # du vol reel : .bin et mocap doivent bien couvrir la meme periode) ────
    df_imu_check = pd.read_csv(imu_csv)
    t_imu_abs_check = df_imu_check["t"].to_numpy()
    overlap_start = max(t_gps_abs[0], t_imu_abs_check[0])
    overlap_end   = min(t_gps_abs[-1], t_imu_abs_check[-1])
    if overlap_end <= overlap_start:
        print("[ERREUR] Aucun recouvrement temporel entre gps_reel.csv et imu_reel.csv !")
        print(f"         GPS/Odom : {t_gps_abs[0]:.2f}s -> {t_gps_abs[-1]:.2f}s")
        print(f"         IMU      : {t_imu_abs_check[0]:.2f}s -> {t_imu_abs_check[-1]:.2f}s")
        print("         Ces fichiers ne correspondent probablement pas au meme vol/segment.")
        sys.exit(1)
    print(f"[FGO REEL] Recouvrement temporel OK : {overlap_start:.2f}s -> {overlap_end:.2f}s "
          f"({overlap_end - overlap_start:.1f}s)")

    # ── Debiaisage robuste sur X, Y ET Z (mediane des N_BIAS premiers points
    # GPS vs Odom) -- identique dans l'esprit a fgo_IMU_GPS_coupled.py, mais
    # directement en metres, sans passer par lat/lon. ──────────────────────
    x0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["x"].to_numpy())) for i in range(N_BIAS)])
    y0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["y"].to_numpy())) for i in range(N_BIAS)])
    z0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["z"].to_numpy())) for i in range(N_BIAS)])
    dx -= np.median(dx[:N_BIAS] - x0)
    dy -= np.median(dy[:N_BIAS] - y0)
    dz -= np.median(dz[:N_BIAS] - z0)

    t0 = df_gps["t"].iloc[0]
    t_gps_full = (df_gps["t"] - t0).to_numpy()

    indices = list(range(0, len(dx), DELTA_K))
    n = len(indices)
    print(f"[FGO REEL] {len(dx)} mesures GPS -> {n} noeuds (DELTA_K={DELTA_K})")

    t_gps = t_gps_full[indices]
    dx, dy, dz = dx[indices], dy[indices], dz[indices]

    # ── Chargement IMU (deja en unites SI) ──────────────────────────────────
    df_imu = pd.read_csv(imu_csv).sort_values("t").reset_index(drop=True)
    t_imu = (df_imu["t"] - t0).to_numpy()
    ax = df_imu["ax"].to_numpy(); ay = df_imu["ay"].to_numpy(); az = df_imu["az"].to_numpy()
    gx = df_imu["wx"].to_numpy(); gy = df_imu["wy"].to_numpy(); gz = df_imu["wz"].to_numpy()

    params = PreintegrationParams.MakeSharedU(GRAVITY)
    params.setAccelerometerCovariance(np.eye(3) * (ACCEL_SIGMA * NOISE_INFLATION_ACCEL)**2)
    params.setGyroscopeCovariance(np.eye(3) * (GYRO_SIGMA * NOISE_INFLATION_GYRO)**2)
    params.setIntegrationCovariance(np.eye(3) * INTEGRATION_COV)

    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    cov_gps        = gtsam.noiseModel.Isotropic.Sigma(3, SIGMA_GPS)
    cov_prior_pose  = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([SIGMA_PRIOR_ROT]*3 + [SIGMA_PRIOR_POS]*3))
    cov_prior_vel   = gtsam.noiseModel.Isotropic.Sigma(3, SIGMA_PRIOR_VEL)
    cov_prior_bias  = gtsam.noiseModel.Isotropic.Sigma(6, SIGMA_PRIOR_BIAS)

    x0_est = np.median(dx[:N_BIAS])
    y0_est = np.median(dy[:N_BIAS])
    z0_est = np.median(dz[:N_BIAS])

    pose0 = Pose3(Rot3(), Point3(x0_est, y0_est, z0_est))
    vel0  = np.zeros(3)
    bias0 = ConstantBias(np.zeros(3), np.zeros(3))

    values.insert(X(0), pose0)
    values.insert(V(0), vel0)
    values.insert(B(0), bias0)

    graph.add(gtsam.PriorFactorPose3(X(0), pose0, cov_prior_pose))
    graph.add(gtsam.PriorFactorVector(V(0), vel0, cov_prior_vel))
    graph.add(gtsam.PriorFactorConstantBias(B(0), bias0, cov_prior_bias))

    for k in range(1, n):
        t_prev, t_k = t_gps[k-1], t_gps[k]
        dt_node = max(t_k - t_prev, 1e-4)

        pim = PreintegratedImuMeasurements(params, bias0)
        mask = (t_imu > t_prev) & (t_imu <= t_k)
        idxs = np.where(mask)[0]
        t_precedent = t_prev
        for idx in idxs:
            dt = max(t_imu[idx] - t_precedent, 1e-4)
            acc  = np.array([ax[idx], ay[idx], az[idx]])
            gyro = np.array([gx[idx], gy[idx], gz[idx]])
            pim.integrateMeasurement(acc, gyro, dt)
            t_precedent = t_imu[idx]

        if k >= 2:
            vel_init = np.array([
                dx[k] - dx[k-2], dy[k] - dy[k-2], dz[k] - dz[k-2]
            ]) / max(t_gps[k] - t_gps[k-2], 1e-3)
        elif k == 1:
            vel_init = np.array([dx[k]-dx[k-1], dy[k]-dy[k-1], dz[k]-dz[k-1]]) / dt_node
        else:
            vel_init = np.zeros(3)

        pose_init = Pose3(Rot3(), Point3(dx[k], dy[k], dz[k]))
        values.insert(X(k), pose_init)
        values.insert(V(k), vel_init)
        values.insert(B(k), bias0)

        graph.add(gtsam.ImuFactor(X(k-1), V(k-1), X(k), V(k), B(k-1), pim))

        sigma_bias_dt = BIAS_RW_SIGMA * np.sqrt(dt_node)
        cov_bias_rw = gtsam.noiseModel.Isotropic.Sigma(6, sigma_bias_dt)
        graph.add(gtsam.BetweenFactorConstantBias(
            B(k-1), B(k), ConstantBias(np.zeros(3), np.zeros(3)), cov_bias_rw))

        graph.add(gtsam.GPSFactor(X(k), Point3(dx[k], dy[k], dz[k]), cov_gps))

    print(f"[FGO REEL] Graphe : {graph.size()} facteurs, {n} noeuds (Pose+Vel+Bias)")

    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(200)
    lm_params.setVerbosity("SILENT")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params)
    t_opt_debut = time.time()
    result = optimizer.optimize()
    duree_opt = time.time() - t_opt_debut
    print(f"[FGO REEL] Temps de calcul (optimisation) : {duree_opt:.4f}s "
          f"({n} noeuds, {graph.size()} facteurs)")

    x_fgo = np.array([result.atPose3(X(k)).translation()[0] for k in range(n)])
    y_fgo = np.array([result.atPose3(X(k)).translation()[1] for k in range(n)])
    z_fgo = np.array([result.atPose3(X(k)).translation()[2] for k in range(n)])

    t_odom = (df_odom["t"] - t0).to_numpy()
    x_odom = np.interp(t_gps, t_odom, df_odom["x"].to_numpy())
    y_odom = np.interp(t_gps, t_odom, df_odom["y"].to_numpy())
    z_odom = np.interp(t_gps, t_odom, df_odom["z"].to_numpy())

    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
    rmse_gps_x, rmse_gps_y, rmse_gps_z = rmse(dx, x_odom), rmse(dy, y_odom), rmse(dz, z_odom)
    rmse_fgo_x, rmse_fgo_y, rmse_fgo_z = rmse(x_fgo, x_odom), rmse(y_fgo, y_odom), rmse(z_fgo, z_odom)

    print(f"[FGO REEL] RMSE GPS synthetique : X={rmse_gps_x:.4f}  Y={rmse_gps_y:.4f}  Z={rmse_gps_z:.4f}")
    print(f"[FGO REEL] RMSE FGO             : X={rmse_fgo_x:.4f}  Y={rmse_fgo_y:.4f}  Z={rmse_fgo_z:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"FGO GPS+IMU couplé — VOL REEL — {nom}  (temps calcul optimisation : {duree_opt:.3f}s)",
                 fontsize=11, fontweight="bold")
    donnees = [
        ("X", dx, x_fgo, x_odom, rmse_gps_x, rmse_fgo_x, "tab:blue"),
        ("Y", dy, y_fgo, y_odom, rmse_gps_y, rmse_fgo_y, "tab:orange"),
        ("Z", dz, z_fgo, z_odom, rmse_gps_z, rmse_fgo_z, "tab:green"),
    ]
    for ax, (label, brut, fgo, verite, r_gps, r_fgo, color) in zip(axes, donnees):
        ax.plot(t_gps, verite, "--", color="black", linewidth=1.3, label=f"{label} vérité (mocap)")
        ax.plot(t_gps, brut, "-", color=color, alpha=0.5, linewidth=1.2, label=f"{label} GPS synthétique")
        ax.plot(t_gps, fgo, "-", color="tab:red", linewidth=1.8, label=f"{label} FGO GPS+IMU")
        ax.text(0.01, 0.95, f"RMSE GPS = {r_gps:.2e} m\nRMSE FGO = {r_fgo:.2e} m",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_ylabel(f"{label} (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True)
    axes[-1].set_xlabel("Temps (s)")
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out_img = os.path.join(IMG_DIR, f"fgo_real_flight_{nom}.pdf")
    plt.savefig(out_img)
    print(f"[FGO REEL] Figure sauvegardée : {out_img}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f"fgo_real_flight_{nom}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "z"])
        writer.writeheader()
        for k in range(n):
            writer.writerow({"t": round(float(t_gps[k]), 4), "x": round(float(x_fgo[k]), 6),
                              "y": round(float(y_fgo[k]), 6), "z": round(float(z_fgo[k]), 6)})
    print(f"[FGO REEL] CSV sauvegardé : {out_csv} ({n} lignes)")

    # ── AJOUT : trajectoire vue du dessus (X vs Y) -- indispensable pour
    # voir clairement un motif "lawnmower" (aller-retours paralleles), pas
    # visible sur les series temporelles X/Y/Z separees ci-dessus. ─────────
    fig_top, ax_top = plt.subplots(figsize=(8, 8))
    ax_top.plot(x_odom, y_odom, "--", color="black", linewidth=1.3, label="Vérité (mocap)")
    ax_top.plot(dx, dy, "-", color="tab:blue", alpha=0.4, linewidth=1.0, label="GPS synthétique")
    ax_top.plot(x_fgo, y_fgo, "-", color="tab:red", linewidth=1.8, label="FGO GPS+IMU")
    ax_top.set_xlabel("X (m)")
    ax_top.set_ylabel("Y (m)")
    ax_top.set_title(f"Trajectoire vue du dessus — VOL REEL — {nom}  "
                      f"(temps calcul optimisation : {duree_opt:.3f}s)", fontweight="bold")
    ax_top.legend(fontsize=9)
    ax_top.grid(True)
    ax_top.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    out_img_top = os.path.join(IMG_DIR, f"fgo_real_flight_{nom}_topdown.pdf")
    plt.savefig(out_img_top)
    print(f"[FGO REEL] Figure vue du dessus sauvegardée : {out_img_top}")

    plt.show()


def main():
    gps_csv  = os.path.join(CSV_DIR, "gps_reel.csv")
    imu_csv  = os.path.join(CSV_DIR, "imu_reel.csv")
    odom_csv = os.path.join(CSV_DIR, "odom_reel.csv")
    for p in (gps_csv, imu_csv, odom_csv):
        if not os.path.isfile(p):
            print(f"[ERREUR] Fichier manquant : {p}")
            print("         Lance d'abord bin_to_imu_csv.py et mocap_to_csv.py.")
            sys.exit(1)
    run_fgo_real(gps_csv, imu_csv, odom_csv)


if __name__ == "__main__":
    main()
