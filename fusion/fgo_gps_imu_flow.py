#!/usr/bin/env python3
"""
fgo_gps_imu_flow.py
--------------------
FGO couple GPS+IMU+Flow Deck en mode batch : un seul graphe de facteurs
GTSAM, extension directe de fgo_IMU_GPS_coupled.py avec deux facteurs
supplementaires par arete entre deux noeuds GPS consecutifs :

  - Facteur ToF (Z absolu) : mesure d'altitude independante de
    l'accelerometre -- motive par la limite identifiee sur le FGO
    GPS+IMU seul (derive en Z sur le vol carre, cf. section coupled du
    rapport) : le GPSFactor ne contraint jamais l'orientation, donc une
    ambiguite accelerometre/inclinaison n'est jamais totalement levee.
    Le ToF, lui, ne depend pas de l'accelerometre.
  - Facteur Flow (X/Y relatif) : somme des deplacements du flux optique
    entre deux noeuds GPS consecutifs, meme logique de fenetre que
    l'ImuFactor (integre TOUT ce qui se passe entre les deux noeuds).

CHOIX D'IMPLEMENTATION IMPORTANT :
  GTSAM n'a pas de facteur natif "position Z seule" ou "translation X/Y
  seule entre deux Pose3". Plutot que d'ecrire des CustomFactor a la
  main (jacobiens manuels, source d'erreurs), on reutilise les facteurs
  generiques deja fournis (GPSFactor, BetweenFactorPose3) en gonflant
  artificiellement le bruit sur les composantes que chaque capteur NE
  mesure PAS :
    - ToF  : GPSFactor avec sigma_x, sigma_y enormes (composantes
      ignorees) et sigma_z reel du capteur -- le residu sur x,y devient
      negligeable quelle que soit la valeur placeholder utilisee.
    - Flow : BetweenFactorPose3 avec sigma_rotation, sigma_z enormes et
      sigma_x, sigma_y reels -- seule la translation horizontale
      relative est effectivement contrainte.
  C'est mathematiquement equivalent a un facteur dedie "partiel", sans
  le risque d'un jacobien manuel incorrect.

Architecture (nœuds, GPSFactor, ImuFactor, biais random-walk, nœud 0) :
  identique a fgo_IMU_GPS_coupled.py -- voir ce fichier pour le detail
  de chaque facteur deja valide sur GPS+IMU seul.

Usage :
  python3 fusion/fgo_gps_imu_flow.py vertical
  python3 fusion/fgo_gps_imu_flow.py               <- detecte toutes les trajectoires
"""

import sys, os
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

X = symbol_shorthand.X   # Pose3 (position + orientation)
V = symbol_shorthand.V   # Vector3 (vitesse)
B = symbol_shorthand.B   # ConstantBias (biais accel+gyro)

_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(_ROOT, "csv_files")
OUT_DIR = os.path.join(CSV_DIR, "fgo_gps_imu_flow_csv")
IMG_DIR = os.path.join(_ROOT, "images")

GRAVITY = 9.8

# ─── Parametres GPS+IMU (identiques a fgo_IMU_GPS_coupled.py) ─────────────
SIGMA_GPS        = 1.0
N_BIAS           = 15
SIGMA_PRIOR_POS  = SIGMA_GPS / np.sqrt(N_BIAS)
SIGMA_PRIOR_ROT  = 0.1
SIGMA_PRIOR_VEL  = 0.1
SIGMA_PRIOR_BIAS = 0.1

ACCEL_SIGMA      = 0.05
GYRO_SIGMA       = 0.02
INTEGRATION_COV  = 1e-8
BIAS_RW_SIGMA    = 5e-4

DELTA_K          = 1

# ─── Parametres Flow Deck ──────────────────────────────────────────────────
HFOV          = 1.047   # champ de vision horizontal (rad) -- cf. SDF
LARGEUR_PX    = 64      # largeur de l'image (pixels) -- cf. SDF
Z_MIN_ECHELLE = 0.05    # borne basse (m) pour l'echelle pixel->metre

SIGMA_TOF   = 0.005     # m -- bruit reel du capteur ToF (SDF)
SIGMA_FLOW  = 0.10      # m -- bruit assume sur le deplacement horizontal
                        # cumule par fenetre (a calibrer empiriquement,
                        # cf. methodologie de debug habituelle)
BIG_SIGMA   = 1.0e4     # m ou rad -- "ignore cette composante"
# ────────────────────────────────────────────────────────────────────────


def gps_to_meters(lat, lon, alt):
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


def charger_flow_tof(flow_csv: str, tof_csv: str, t0: float):
    """
    Charge flow + tof, nettoie le ToF (inf -> interpolation), convertit
    le flux optique en metres (echelle pixel->metre via la hauteur ToF
    courante). Retourne (t_flow, dx_m, dy_m, z_tof) alignes sur les
    temps du flow.
    """
    df_flow = pd.read_csv(flow_csv).sort_values("t").reset_index(drop=True)
    df_tof  = pd.read_csv(tof_csv).sort_values("t").reset_index(drop=True)

    t_flow = (df_flow["t"] - t0).to_numpy()
    t_tof  = (df_tof["t"] - t0).to_numpy()

    range_tof_brut = df_tof["range"].to_numpy()
    valide = np.isfinite(range_tof_brut)
    if not valide.all():
        n_invalides = int((~valide).sum())
        print(f"[FGO GPS+IMU+Flow] {n_invalides} valeurs ToF non finies nettoyées par interpolation")
        range_tof = np.interp(t_tof, t_tof[valide], range_tof_brut[valide])
    else:
        range_tof = range_tof_brut

    z_flow = np.interp(t_flow, t_tof, range_tof)
    z_pour_echelle = np.clip(z_flow, Z_MIN_ECHELLE, None)

    dx_px = df_flow["dx"].to_numpy()
    dy_px = df_flow["dy"].to_numpy()
    echelle = (2.0 * z_pour_echelle * np.tan(HFOV / 2.0)) / LARGEUR_PX
    dx_m = dx_px * echelle
    dy_m = dy_px * echelle

    return t_flow, dx_m, dy_m, z_flow, t_tof, range_tof


def detecter_trajectoires():
    import re
    trajectoires = {}
    for f in os.listdir(CSV_DIR):
        m = re.match(r"^gps(?:_(.+))?\.csv$", f)
        if m:
            nom  = m.group(1) or "sol"
            suf  = f"_{nom}" if nom != "sol" else ""
            gps  = os.path.join(CSV_DIR, f"gps{suf}.csv")
            imu  = os.path.join(CSV_DIR, f"imu{suf}.csv")
            flow = os.path.join(CSV_DIR, f"flow{suf}.csv")
            tof  = os.path.join(CSV_DIR, f"tof{suf}.csv")
            odom = os.path.join(CSV_DIR, f"odom{suf}.csv")
            if all(os.path.isfile(p) for p in [gps, imu, flow, tof, odom]):
                trajectoires[nom] = {"gps": gps, "imu": imu, "flow": flow, "tof": tof, "odom": odom}
    return trajectoires


def run_fgo_gps_imu_flow(nom: str, gps_csv: str, imu_csv: str, flow_csv: str, tof_csv: str, odom_csv: str):
    print(f"\n[FGO GPS+IMU+Flow] Trajectoire : {nom}")

    # ── GPS ──────────────────────────────────────────────────────────────
    df_gps = pd.read_csv(gps_csv).sort_values("t").reset_index(drop=True)
    dx, dy, dz = gps_to_meters(df_gps["lat"].to_numpy(), df_gps["lon"].to_numpy(),
                                df_gps["alt"].to_numpy())

    df_odom = pd.read_csv(odom_csv).sort_values("t").reset_index(drop=True)
    t_od_ref  = df_odom["t"].to_numpy()
    t_gps_abs = df_gps["t"].to_numpy()
    x_off = float(np.interp(t_gps_abs[0], t_od_ref, df_odom["x"].to_numpy()))
    y_off = float(np.interp(t_gps_abs[0], t_od_ref, df_odom["y"].to_numpy()))
    z_off = float(np.interp(t_gps_abs[0], t_od_ref, df_odom["z"].to_numpy()))
    dx += x_off; dy += y_off; dz += z_off

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
    print(f"[FGO GPS+IMU+Flow] {len(dx)} mesures GPS -> {n} noeuds (DELTA_K={DELTA_K})")

    t_gps = t_gps_full[indices]
    dx, dy, dz = dx[indices], dy[indices], dz[indices]

    # ── IMU ──────────────────────────────────────────────────────────────
    df_imu = pd.read_csv(imu_csv).sort_values("t").reset_index(drop=True)
    t_imu = (df_imu["t"] - t0).to_numpy()
    ax = df_imu["ax"].to_numpy(); ay = df_imu["ay"].to_numpy(); az = df_imu["az"].to_numpy()
    if all(c in df_imu.columns for c in ("wx", "wy", "wz")):
        gx = df_imu["wx"].to_numpy(); gy = df_imu["wy"].to_numpy(); gz = df_imu["wz"].to_numpy()
    else:
        print("[FGO GPS+IMU+Flow] ATTENTION : colonnes wx/wy/wz absentes -> gyro mis a zero")
        gx = gy = gz = np.zeros_like(ax)

    # ── Flow + ToF ───────────────────────────────────────────────────────
    t_flow, dx_flow_m, dy_flow_m, z_flow_interp, t_tof, range_tof = charger_flow_tof(flow_csv, tof_csv, t0)

    params = PreintegrationParams.MakeSharedU(GRAVITY)
    params.setAccelerometerCovariance(np.eye(3) * ACCEL_SIGMA**2)
    params.setGyroscopeCovariance(np.eye(3) * GYRO_SIGMA**2)
    params.setIntegrationCovariance(np.eye(3) * INTEGRATION_COV)

    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    cov_gps         = gtsam.noiseModel.Isotropic.Sigma(3, SIGMA_GPS)
    cov_prior_pose   = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([SIGMA_PRIOR_ROT]*3 + [SIGMA_PRIOR_POS]*3))
    cov_prior_vel    = gtsam.noiseModel.Isotropic.Sigma(3, SIGMA_PRIOR_VEL)
    cov_prior_bias   = gtsam.noiseModel.Isotropic.Sigma(6, SIGMA_PRIOR_BIAS)

    # Facteur ToF : GPSFactor "detourne", sigma x/y enormes -> seul Z compte
    cov_tof = gtsam.noiseModel.Diagonal.Sigmas(np.array([BIG_SIGMA, BIG_SIGMA, SIGMA_TOF]))

    # Facteur Flow : BetweenFactorPose3 "detourne", sigma rot/z enormes ->
    # seules les translations x,y comptent
    cov_flow = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([BIG_SIGMA, BIG_SIGMA, BIG_SIGMA, SIGMA_FLOW, SIGMA_FLOW, BIG_SIGMA]))

    # ── Noeud 0 : prior (median des N_BIAS premiers points GPS) ───────────
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

    # ── Noeuds 1..n-1 : GPS + ImuFactor + biais random-walk + ToF + Flow ──
    for k in range(1, n):
        t_prev, t_k = t_gps[k-1], t_gps[k]
        dt_node = max(t_k - t_prev, 1e-4)

        # -- ImuFactor (inchange) -------------------------------------------
        pim = PreintegratedImuMeasurements(params, bias0)
        mask_imu = (t_imu > t_prev) & (t_imu <= t_k)
        idxs_imu = np.where(mask_imu)[0]
        t_precedent = t_prev
        for idx in idxs_imu:
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

        # -- Facteur ToF (Z absolu) : moyenne des lectures ToF de la fenêtre
        mask_tof = (t_tof > t_prev) & (t_tof <= t_k)
        idxs_tof = np.where(mask_tof)[0]
        if len(idxs_tof) > 0:
            z_tof_mean = float(np.mean(range_tof[idxs_tof]))
            graph.add(gtsam.GPSFactor(X(k), Point3(0.0, 0.0, z_tof_mean), cov_tof))

        # -- Facteur Flow (X/Y relatif) : somme des deplacements de la fenêtre
        mask_flow = (t_flow > t_prev) & (t_flow <= t_k)
        idxs_flow = np.where(mask_flow)[0]
        if len(idxs_flow) > 0:
            dx_sum = float(np.sum(dx_flow_m[idxs_flow]))
            dy_sum = float(np.sum(dy_flow_m[idxs_flow]))
            deplacement_flow = Pose3(Rot3(), Point3(dx_sum, dy_sum, 0.0))
            graph.add(gtsam.BetweenFactorPose3(X(k-1), X(k), deplacement_flow, cov_flow))

    print(f"[FGO GPS+IMU+Flow] Graphe : {graph.size()} facteurs, {n} noeuds (Pose+Vel+Bias)")

    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(200)
    lm_params.setVerbosity("SILENT")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params)
    result = optimizer.optimize()

    print(f"[FGO GPS+IMU+Flow] Erreur initiale : {graph.error(values):.4f}")
    print(f"[FGO GPS+IMU+Flow] Erreur finale   : {graph.error(result):.4f}")

    x_fgo = np.array([result.atPose3(X(k)).translation()[0] for k in range(n)])
    y_fgo = np.array([result.atPose3(X(k)).translation()[1] for k in range(n)])
    z_fgo = np.array([result.atPose3(X(k)).translation()[2] for k in range(n)])

    # ── RMSE vs Odom et vs GPS brut ─────────────────────────────────────
    t_odom = (df_odom["t"] - t0).to_numpy()
    x_odom = np.interp(t_gps, t_odom, df_odom["x"].to_numpy())
    y_odom = np.interp(t_gps, t_odom, df_odom["y"].to_numpy())
    z_odom = np.interp(t_gps, t_odom, df_odom["z"].to_numpy())

    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
    rmse_gps_x, rmse_gps_y, rmse_gps_z = rmse(dx, x_odom), rmse(dy, y_odom), rmse(dz, z_odom)
    rmse_fgo_x, rmse_fgo_y, rmse_fgo_z = rmse(x_fgo, x_odom), rmse(y_fgo, y_odom), rmse(z_fgo, z_odom)

    print(f"[FGO GPS+IMU+Flow] RMSE GPS brut : X={rmse_gps_x:.4f}  Y={rmse_gps_y:.4f}  Z={rmse_gps_z:.4f}")
    print(f"[FGO GPS+IMU+Flow] RMSE FGO      : X={rmse_fgo_x:.4f}  Y={rmse_fgo_y:.4f}  Z={rmse_fgo_z:.4f}")

    # ── Tracé ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"FGO GPS+IMU+Flow couplé — {nom}", fontsize=11, fontweight="bold")

    donnees = [
        ("X — Est/Ouest", dx, x_fgo, x_odom, rmse_gps_x, rmse_fgo_x, "tab:blue"),
        ("Y — Nord/Sud",  dy, y_fgo, y_odom, rmse_gps_y, rmse_fgo_y, "tab:orange"),
        ("Z — Altitude",  dz, z_fgo, z_odom, rmse_gps_z, rmse_fgo_z, "tab:green"),
    ]
    for ax, (label, brut, fgo, verite, r_gps, r_fgo, color) in zip(axes, donnees):
        ax.plot(t_gps, verite, "--", color="black", linewidth=1.3, label=f"{label} vérité (Odom)")
        ax.plot(t_gps, brut, "-", color=color, alpha=0.5, linewidth=1.2, label=f"{label} GPS brut")
        ax.plot(t_gps, fgo, "-", color="tab:red", linewidth=1.8, label=f"{label} FGO GPS+IMU+Flow")
        ax.text(0.01, 0.95, f"RMSE GPS = {r_gps:.2e} m\nRMSE FGO = {r_fgo:.2e} m",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_ylabel(f"{label} (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True)

    axes[-1].set_xlabel("Temps (s)")
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out_img = os.path.join(IMG_DIR, f"fgo_gps_imu_flow_{nom}.pdf")
    plt.savefig(out_img)
    print(f"[FGO GPS+IMU+Flow] Figure sauvegardée : {out_img}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f"fgo_gps_imu_flow_{nom}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "z"])
        writer.writeheader()
        for k in range(n):
            writer.writerow({
                "t": round(float(t_gps[k]), 4),
                "x": round(float(x_fgo[k]), 6),
                "y": round(float(y_fgo[k]), 6),
                "z": round(float(z_fgo[k]), 6),
            })
    print(f"[FGO GPS+IMU+Flow] CSV sauvegardé : {out_csv} ({n} lignes)")


def main():
    trajectoires = detecter_trajectoires()
    if not trajectoires:
        print("[Erreur] Aucun groupe GPS/IMU/Flow/ToF/Odom trouvé dans csv_files/")
        return

    if len(sys.argv) > 1:
        nom = sys.argv[1]
        if nom not in trajectoires:
            print(f"[Erreur] Trajectoire '{nom}' introuvable. "
                  f"Disponibles : {list(trajectoires.keys())}")
            return
        trajectoires = {nom: trajectoires[nom]}

    for nom, f in trajectoires.items():
        run_fgo_gps_imu_flow(nom, f["gps"], f["imu"], f["flow"], f["tof"], f["odom"])

    plt.show()


if __name__ == "__main__":
    main()