#!/usr/bin/env python3
"""
fgo_uncoupled.py -- TRUE "uncoupled" GPS/IMU estimator.

PROFESSOR'S CORRECTION (followed here):
  "Uncoupled" does NOT mean running a factor graph on the GPS side.
  Optimizing/fusing implies combining >= 2 sources of information --
  with GPS alone, there is nothing to combine, so building a GTSAM
  graph (Prior + Between factors) for GPS was an unnecessary, wrong
  complication. The correct uncoupled logic is simply:

      IF GPS available:      output = raw GPS measurement (no processing)
      IF GPS NOT available:  output = IMU dead-reckoning (predict())

  No factor graph, no optimization, anywhere on the GPS side. The IMU
  side still uses GTSAM's preintegration (PreintegratedImuMeasurements
  + predict()), because that's just numerical integration of raw
  sensor data, not an optimization/fusion step.

Real orientation and real velocity (from Gazebo's Odom) are used as
the IMU's starting state during each outage -- this was requested by
the professor specifically to isolate whether the remaining drift
comes from IMU noise itself, independent of orientation/velocity
estimation errors.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gtsam import PreintegrationParams, PreintegratedImuMeasurements, NavState, Rot3, Point3
from gtsam.imuBias import ConstantBias

_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(_ROOT, "csv_files")
OUT_DIR = os.path.join(CSV_DIR, "uncoupled_csv")
IMG_DIR = os.path.join(_ROOT, "images")

GRAVITY = 9.8
ACCEL_SIGMA, GYRO_SIGMA, INTEGRATION_COV = 0.0173, 0.02, 1e-8  # assumed IMU noise (tactical grade)
SEUIL_COUPURE_S = 0.3     # gap between 2 GPS timestamps above this = real outage (not sampling jitter)
USE_REAL_V0 = True        # True: use real Gazebo velocity as v0. False: 2-point finite difference.


def gps_to_meters(lat, lon, alt):
    """Convert lat/lon/alt (NavSat) to local flat-earth meters,
    referenced to the median of the first few samples."""
    R, N_REF = 6_371_000.0, min(10, len(lat))
    lat_rad = np.radians(lat)
    lat0, lon0, alt0 = np.median(lat_rad[:N_REF]), np.median(lon[:N_REF]), np.median(alt[:N_REF])
    dx = (lon - lon0) * np.cos(lat0) * np.radians(R)
    dy = (lat - np.median(lat[:N_REF])) * np.radians(R)
    return dx, dy, alt - alt0


def detecter_trajectoires():
    """Find every (gps, imu, odom) CSV triplet available in csv_files/."""
    import re
    trajectoires = {}
    for f in os.listdir(CSV_DIR):
        m = re.match(r"^gps(?:_(.+))?\.csv$", f)
        if m:
            nom = m.group(1) or "sol"
            suf = f"_{nom}" if nom != "sol" else ""
            paths = {k: os.path.join(CSV_DIR, f"{k}{suf}.csv") for k in ("gps", "imu", "odom")}
            if all(os.path.isfile(p) for p in paths.values()):
                trajectoires[nom] = paths
    return trajectoires


def detecter_coupures_reelles(t_gps, t_imu_fin=None):
    """A real GPS outage = a gap between two consecutive GPS timestamps
    larger than SEUIL_COUPURE_S (not just normal 10Hz jitter).

    Also detects a FINAL outage that never ends (GPS cut for good and
    never comes back, e.g. Bloc C in read_gps.py: GPS only 0-3s then
    nothing). In that case there is no "next GPS point" to close the
    window on, so the window is bounded by the end of the IMU log
    instead (t_imu_fin)."""
    ecarts = np.diff(t_gps)
    fenetres = [(t_gps[i], t_gps[i+1]) for i, e in enumerate(ecarts) if e > SEUIL_COUPURE_S]
    if t_imu_fin is not None and (t_imu_fin - t_gps[-1]) > SEUIL_COUPURE_S:
        fenetres.append((t_gps[-1], t_imu_fin))
    return fenetres


def run_uncoupled(nom, gps_csv, imu_csv, odom_csv):
    print(f"\n[Uncoupled] Trajectoire : {nom}")

    # ============================================================
    # LOAD DATA
    # ============================================================

    # ---- GPS: raw lat/lon/alt -> local meters, re-anchored to match
    # Odom's frame at t=0 (needed only to compare against ground truth
    # on the same coordinate origin -- this is NOT a fusion/optimization
    # step, just a coordinate conversion) ----
    df_gps = pd.read_csv(gps_csv).sort_values("t").reset_index(drop=True)
    dx, dy, dz = gps_to_meters(df_gps["lat"].to_numpy(), df_gps["lon"].to_numpy(), df_gps["alt"].to_numpy())

    df_odom = pd.read_csv(odom_csv).sort_values("t").reset_index(drop=True)
    t_od_ref = df_odom["t"].to_numpy()
    t_gps_abs = df_gps["t"].to_numpy()
    for arr, col in ((dx, "x"), (dy, "y"), (dz, "z")):
        arr += float(np.interp(t_gps_abs[0], t_od_ref, df_odom[col].to_numpy()))

    # ---- Constant offset correction (median over first N_BIAS points
    # vs ground truth) -- this is just removing a coordinate-frame
    # offset, not smoothing/optimizing individual GPS points. Every
    # point keeps its own raw noise afterward. ----
    N_BIAS = 5
    x0 = np.array([np.interp(t_gps_abs[i], t_od_ref, df_odom["x"].to_numpy()) for i in range(N_BIAS)])
    y0 = np.array([np.interp(t_gps_abs[i], t_od_ref, df_odom["y"].to_numpy()) for i in range(N_BIAS)])
    z0 = np.array([np.interp(t_gps_abs[i], t_od_ref, df_odom["z"].to_numpy()) for i in range(N_BIAS)])
    dx -= np.mean(dx[:N_BIAS] - x0)
    dy -= np.mean(dy[:N_BIAS] - y0)
    dz -= np.mean(dz[:N_BIAS] - z0)  # FIX: was missing -- same correction as x,y, applied to z

    t0 = df_gps["t"].iloc[0]
    t_gps = (df_gps["t"] - t0).to_numpy()
    t_odom_rel = (df_odom["t"] - t0).to_numpy()

    # ---- Real orientation lookup (Odom quaternion, nearest sample) ----
    quat_ok = all(c in df_odom.columns for c in ("qw", "qx", "qy", "qz"))
    if quat_ok:
        qw, qx, qy, qz = (df_odom[c].to_numpy() for c in ("qw", "qx", "qy", "qz"))
    def rotation_reelle_a(t):
        if not quat_ok:
            return Rot3()  # fallback: identity (flat) if not logged
        idx = int(np.argmin(np.abs(t_odom_rel - t)))
        return Rot3.Quaternion(qw[idx], qx[idx], qy[idx], qz[idx])

    # ---- Real velocity lookup (Odom twist.linear, world frame,
    # published directly by Gazebo -- verified to match finite
    # difference on ground-truth position) ----
    vel_ok = all(c in df_odom.columns for c in ("vx", "vy", "vz"))
    if vel_ok:
        vx, vy, vz = (df_odom[c].to_numpy() for c in ("vx", "vy", "vz"))
    def v0_reel_a(t):
        return np.array([np.interp(t, t_odom_rel, vx), np.interp(t, t_odom_rel, vy), np.interp(t, t_odom_rel, vz)])

    # ---- IMU: raw accelerometer + gyroscope ----
    df_imu = pd.read_csv(imu_csv).sort_values("t").reset_index(drop=True)
    t_imu = (df_imu["t"] - t0).to_numpy()
    ax, ay, az = (df_imu[c].to_numpy() for c in ("ax", "ay", "az"))
    if all(c in df_imu.columns for c in ("wx", "wy", "wz")):
        gx, gy, gz = (df_imu[c].to_numpy() for c in ("wx", "wy", "wz"))
    else:
        gx = gy = gz = np.zeros_like(ax)

    fenetres = detecter_coupures_reelles(t_gps, t_imu_fin=t_imu[-1])
    print(f"[Uncoupled] {len(fenetres)} coupure(s) detectee(s) : {fenetres}")

    # ---- IMU preintegration parameters (raw numerical integration
    # config, NOT an optimizer/fusion setting) ----
    params_imu = PreintegrationParams.MakeSharedU(GRAVITY)
    params_imu.setAccelerometerCovariance(np.eye(3) * ACCEL_SIGMA**2)
    params_imu.setGyroscopeCovariance(np.eye(3) * GYRO_SIGMA**2)
    params_imu.setIntegrationCovariance(np.eye(3) * INTEGRATION_COV)
    bias_hat = ConstantBias(np.zeros(3), np.zeros(3))  # bias fixed at 0, never estimated in uncoupled

    resultats = []

    # ============================================================
    # STEP 1 -- GPS AVAILABLE: output the RAW measurement directly.
    # NO gtsam.NonlinearFactorGraph here. NO Prior/Between factors.
    # This is exactly "print GPS output" as requested.
    # ============================================================
    idxs_gps_hors_coupure = np.ones(len(t_gps), dtype=bool)
    for fdeb, ffin in fenetres:
        idxs_gps_hors_coupure &= ~((t_gps > fdeb) & (t_gps < ffin))
    for k in np.where(idxs_gps_hors_coupure)[0]:
        resultats.append((t_gps[k], dx[k], dy[k], dz[k], "GPS"))

    # ============================================================
    # STEP 2 -- GPS NOT AVAILABLE (outage): dead-reckon with the IMU.
    # Starting position = the last RAW GPS point (no smoothing, no
    # regression -- just the raw measurement, per the professor's
    # "simple, no extra steps" instruction).
    # predict() is plain numerical integration, not an optimizer.
    # ============================================================
    for fdeb, ffin in fenetres:
        idxs_avant = np.where(t_gps < fdeb)[0]
        if len(idxs_avant) == 0:
            continue
        k_prev = idxs_avant[-1]
        t_prev = t_gps[k_prev]
        xp, yp, zp = dx[k_prev], dy[k_prev], dz[k_prev]  # raw GPS point, unmodified

        # Fallback velocity (used only if USE_REAL_V0 is False, or if
        # real velocity isn't logged): simple 2-point finite difference.
        if len(idxs_avant) >= 2:
            k_prev2 = idxs_avant[-2]
            v0_diff = np.array([xp - dx[k_prev2], yp - dy[k_prev2], zp - dz[k_prev2]]) / max(t_prev - t_gps[k_prev2], 1e-3)
        else:
            v0_diff = np.zeros(3)
        v0 = v0_reel_a(t_prev) if (USE_REAL_V0 and vel_ok) else v0_diff

        rot0 = rotation_reelle_a(t_prev)
        print(f"[Uncoupled] t={t_prev:.2f}s  raw GPS start=({xp:.3f},{yp:.3f},{zp:.3f})  v0={v0}")

        # NavState = (orientation, position, velocity) at the start of
        # the outage window -- input to IMU preintegration.
        initial_state = NavState(rot0, Point3(xp, yp, zp), v0)
        pim = PreintegratedImuMeasurements(params_imu, bias_hat)
        idxs_imu = np.where((t_imu >= fdeb) & (t_imu <= ffin))[0]
        t_prec = fdeb
        for k in idxs_imu:
            dt = max(t_imu[k] - t_prec, 1e-4)
            pim.integrateMeasurement(np.array([ax[k], ay[k], az[k]]), np.array([gx[k], gy[k], gz[k]]), dt)
            p = pim.predict(initial_state, bias_hat).position()  # pure integration, reconstructed at each sample
            resultats.append((t_imu[k], p[0], p[1], p[2], "IMU"))
            t_prec = t_imu[k]

    # ============================================================
    # ASSEMBLE, EVALUATE, PLOT, SAVE
    # ============================================================
    resultats.sort(key=lambda r: r[0])
    t_res, x_res, y_res, z_res = (np.array([r[i] for r in resultats]) for i in range(4))
    modes = np.array([r[4] for r in resultats])

    t_odom = (df_odom["t"] - t0).to_numpy()
    x_odom = np.interp(t_res, t_odom, df_odom["x"].to_numpy())
    y_odom = np.interp(t_res, t_odom, df_odom["y"].to_numpy())
    z_odom = np.interp(t_res, t_odom, df_odom["z"].to_numpy())

    rmse = lambda a, b: np.sqrt(np.mean((a - b)**2))
    mask_gps, mask_imu = modes == "GPS", modes == "IMU"

    print(f"[Uncoupled] RMSE GPS : X={rmse(x_res[mask_gps],x_odom[mask_gps]):.4f}  "
          f"Y={rmse(y_res[mask_gps],y_odom[mask_gps]):.4f}  Z={rmse(z_res[mask_gps],z_odom[mask_gps]):.4f}")
    print(f"[Uncoupled] RMSE IMU : X={rmse(x_res[mask_imu],x_odom[mask_imu]):.4f}  "
          f"Y={rmse(y_res[mask_imu],y_odom[mask_imu]):.4f}  Z={rmse(z_res[mask_imu],z_odom[mask_imu]):.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Estimateur GPS/IMU non-couple (raw + erreur, sans FGO) — {nom}", fontsize=11, fontweight="bold")
    for ax, (label, est, verite, color) in zip(axes, [
        ("X", x_res, x_odom, "tab:blue"), ("Y", y_res, y_odom, "tab:orange"), ("Z", z_res, z_odom, "tab:green")]):
        erreur = est - verite  # third trace: error (estimate - truth), same convention as the other validation plots
        ax.plot(t_res, verite, "--", color="black", linewidth=1.3, label=f"{label} vérité")
        segs_gps_done = segs_imu_done = False
        i0 = 0
        while i0 < len(t_res):
            i1 = i0
            while i1+1 < len(t_res) and modes[i1+1] == modes[i0]:
                i1 += 1
            is_gps = modes[i0] == "GPS"
            style = ("-", color) if is_gps else ("-.", "tab:red")
            lbl = None
            if is_gps and not segs_gps_done:
                lbl = f"{label} GPS (raw)"; segs_gps_done = True
            elif not is_gps and not segs_imu_done:
                lbl = f"{label} IMU (coupure)"; segs_imu_done = True
            ax.plot(t_res[i0:i1+1], est[i0:i1+1], style[0], color=style[1], linewidth=1.5, label=lbl)
            i0 = i1 + 1
        # erreur (estime - verite), meme convention que "mesuré/vérité/erreur" des autres figures
        ax.plot(t_res, erreur, ":", color=color, linewidth=1.0, alpha=0.7, label=f"{label} erreur (estimé-vérité)")
        for fdeb, ffin in fenetres:
            ax.axvspan(fdeb, ffin, facecolor="gray", alpha=0.12)  # grey = outage window
        r_gps = rmse(est[mask_gps], verite[mask_gps]) if mask_gps.any() else float("nan")
        r_imu = rmse(est[mask_imu], verite[mask_imu]) if mask_imu.any() else float("nan")
        ax.text(0.01, 0.95, f"RMSE {label} (GPS) = {r_gps:.2e} m\nRMSE {label} (IMU) = {r_imu:.2e} m",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_ylabel(f"{label} (m)"); ax.legend(fontsize=8); ax.grid(True)
    axes[-1].set_xlabel("Temps (s)")
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMG_DIR, f"uncoupled_{nom}.pdf"))
    print(f"[Uncoupled] Figure sauvegardée")

    # ---- DIAGNOSTIC FIGURE: pure IMU drift, offset removed ----
    # Each IMU segment mixes (1) the starting-point offset (raw GPS
    # noise), (2) the TRUE motion of the drone during the outage (e.g.
    # a vertical flight climbs then descends -- not a straight line),
    # and (3) the drift from integrating IMU noise twice. We want (3)
    # only. Subtracting a 2-point straight line removes (1) but wrongly
    # keeps/distorts (2) whenever the true trajectory isn't linear
    # (typically visible on Z). Instead we subtract the TRUE trajectory
    # (ground-truth Odom, already interpolated onto t_res as x/y/z_odom)
    # point by point, then remove only the constant starting offset so
    # every segment starts at 0 -- what remains is purely the noise-
    # driven curvature (2), independent of (1) and (2)true-motion.
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig2.suptitle(f"Dérive IMU pure (offset retiré) — {nom}", fontsize=11, fontweight="bold")
    for axi, (label, color, res_col, verite_col) in enumerate(zip(
            ["X", "Y", "Z"], ["tab:blue", "tab:orange", "tab:green"],
            [x_res, y_res, z_res], [x_odom, y_odom, z_odom])):
        ax2 = axes2[axi]
        i0 = 0
        while i0 < len(t_res):
            i1 = i0
            while i1+1 < len(t_res) and modes[i1+1] == modes[i0]:
                i1 += 1
            if modes[i0] == "IMU" and i1 > i0:
                t_seg = t_res[i0:i1+1]
                v_seg = res_col[i0:i1+1]
                verite_seg = verite_col[i0:i1+1]
                ecart = v_seg - verite_seg          # error vs. true motion (removes true trajectory shape)
                derive = ecart - ecart[0]            # remove only the constant starting offset
                ax2.plot(t_seg, derive, "-o", color=color, markersize=3, linewidth=1.2)
            i0 = i1 + 1
        ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax2.set_ylabel(f"{label} drift (m)"); ax2.grid(True)
    axes2[-1].set_xlabel("Temps (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"uncoupled_{nom}_derive_pure.pdf"))
    print(f"[Uncoupled] Figure dérive pure sauvegardée")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"uncoupled_{nom}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "x", "y", "z", "mode"])
        w.writeheader()
        for i in range(len(t_res)):
            w.writerow({"t": round(float(t_res[i]),4), "x": round(float(x_res[i]),6),
                        "y": round(float(y_res[i]),6), "z": round(float(z_res[i]),6), "mode": modes[i]})
    print(f"[Uncoupled] CSV sauvegardé ({len(t_res)} lignes)")


def main():
    trajectoires = detecter_trajectoires()
    if not trajectoires:
        print("[Erreur] Aucun groupe GPS/IMU/Odom trouvé")
        return
    if len(sys.argv) > 1:
        nom = sys.argv[1]
        if nom not in trajectoires:
            print(f"[Erreur] '{nom}' introuvable. Disponibles : {list(trajectoires.keys())}")
            return
        trajectoires = {nom: trajectoires[nom]}
    for nom, f in trajectoires.items():
        run_uncoupled(nom, f["gps"], f["imu"], f["odom"])
    plt.show()


if __name__ == "__main__":
    main()
!/usr/bin/env python3
