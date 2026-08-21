# #!/usr/bin/env python3
# """
# fgo_IMU_GPS_coupled.py
# --------------
# FGO couple GPS+IMU en mode batch : un seul graphe de facteurs GTSAM,
# avec preintegration IMU comme facteur (ImuFactor) reliant deux noeuds
# consecutifs, facteur GPS (GPSFactor) sur chaque noeud ou une mesure GPS
# existe, et le biais IMU comme variable estimee (marche aleatoire lente).

# Architecture (cf. papier Cioaca et al., Fig. 2 -- "loosely-coupled
# GNSS/IMU factor graph") :
#   - Noeuds : Pose3 (position+orientation), vitesse, biais, un triplet
#     par instant GPS recu.
#   - Facteur Prior : ancre le premier noeud (pose, vitesse, biais).
#   - Facteur GPS : contrainte absolue sur la position (ignore
#     l'orientation), a chaque instant ou une mesure GPS existe.
#   - Facteur ImuFactor : relie 2 noeuds consecutifs via la preintegration
#     de TOUS les echantillons IMU entre les deux -- y compris pendant
#     une coupure GPS, ou cette fenetre devient simplement plus longue,
#     sans casser le graphe.
#   - Facteur Between sur le biais : modelise sa derive lente (marche
#     aleatoire), permettant a l'optimiseur de le corriger au fil du vol.
#     La covariance de ce facteur est mise a l'echelle de sqrt(dt) --
#     coherent avec un modele de bruit blanc integre (marche aleatoire
#     continue) : Var(b_k - b_{k-1}) = q * dt.

# Contrairement a l'estimateur "uncoupled", il n'y a ICI qu'UN SEUL
# probleme d'optimisation global sur tout le vol : GPS et IMU contribuent
# ensemble au meme cout, et se corrigent mutuellement.

# CORRECTIONS APPLIQUEES (par rapport a la version precedente) :
#   1. Correction de biais sur Z (median robuste sur les N_BIAS premieres
#      mesures), en plus de X/Y deja corriges -- evite l'ancrage sur un
#      premier point GPS isole potentiellement bruite.
#   2. Covariance du biais random-walk mise a l'echelle de sqrt(dt) a
#      chaque iteration, au lieu d'une constante fixe.
#   3. Vitesse initiale par difference finie sur les positions GPS DEJA
#      DEBIAISEES (pas l'Odom, qui ne serait jamais disponible en vrai
#      scenario de vol) -- meilleur point de depart pour l'optimiseur
#      non-lineaire que vel_init=0 partout.
#   4. DELTA_K rendu reglable (defaut=1, comportement inchange) pour
#      permettre de tester l'effet du sous-echantillonnage sur le RMSE.
#   5. Noeud 0 ancre sur la mediane des N_BIAS premiers points GPS
#      (pas le premier point brut isole) + SIGMA_PRIOR_POS coherent
#      statistiquement avec cet estimateur + GPSFactor redondant sur
#      X(0) retire.
#   6. AJOUT : trace des 6 composantes du biais IMU estime (accel x,y,z
#      + gyro x,y,z) au fil du vol -- demande du prof, pour verifier
#      visuellement qu'elles convergent.

# Usage :
#   python3 fusion/fgo_IMU_GPS_coupled.py vertical
#   python3 fusion/fgo_IMU_GPS_coupled.py               <- detecte toutes les trajectoires
# """

# import sys, os, time
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# import csv
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import gtsam
# from gtsam import symbol_shorthand
# from gtsam import PreintegrationParams, PreintegratedImuMeasurements
# from gtsam import Pose3, Rot3, Point3
# from gtsam.imuBias import ConstantBias

# X = symbol_shorthand.X   # Pose3 (position + orientation)
# V = symbol_shorthand.V   # Vector3 (vitesse)
# B = symbol_shorthand.B   # ConstantBias (biais accel+gyro)

# _ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# CSV_DIR = os.path.join(_ROOT, "csv_files")
# OUT_DIR = os.path.join(CSV_DIR, "fgo_imu_gps_coupled_csv")
# IMG_DIR = os.path.join(_ROOT, "images")

# GRAVITY = 9.8

# # ─── Parametres modifiables ───────────────────────────────────────────────
# SIGMA_GPS        = 1.0     # m       -- coherent avec le bruit SDF homogene
# N_BIAS           = 15      # nb de mesures GPS utilisees pour le debiaisage initial
# SIGMA_PRIOR_POS  = SIGMA_GPS / np.sqrt(N_BIAS)  # ~0.258 m -- coherent avec la
#                            # precision statistique d'une mediane sur N_BIAS points
#                            # (au lieu d'une valeur fixe arbitraire 10x trop confiante)
# SIGMA_PRIOR_ROT  = 0.1     # rad
# SIGMA_PRIOR_VEL  = 0.1     # m/s
# SIGMA_PRIOR_BIAS = 0.1     # -- incertitude initiale sur le biais

# ACCEL_SIGMA      = 0.173    # m/s^2   -- bruit reel du SDF (accelerometre)
# GYRO_SIGMA       = 0.02    # rad/s   -- bruit reel du SDF (gyroscope)
# INTEGRATION_COV  = 1e-8
# BIAS_RW_SIGMA    = 5e-3    # 5e-3 taux de derive du biais (sigma "par racine de seconde")
#                            # -- teste x10 : 5e-4 figeait le biais sur vol "carre" (62s)
#                            # -- la covariance reelle utilisee = BIAS_RW_SIGMA * sqrt(dt)

# DELTA_K          = 1       # sous-echantillonnage GPS : 1 noeud tous les DELTA_K points
#                            # (1 = comportement d'origine, aucun sous-echantillonnage)
# # ────────────────────────────────────────────────────────────────────────


# def gps_to_meters(lat, lon, alt):
#     """Identique aux scripts precedents : conversion lat/lon/alt -> metres."""
#     R     = 6_371_000.0
#     N_REF = min(10, len(lat))
#     lat_rad = np.radians(lat)
#     lat0  = np.median(lat_rad[:N_REF])
#     lon0  = np.median(lon[:N_REF])
#     alt0  = np.median(alt[:N_REF])
#     dx = (lon - lon0) * np.cos(lat0) * np.radians(R)
#     dy = (lat - np.median(lat[:N_REF])) * np.radians(R)
#     dz = alt - alt0
#     return dx, dy, dz


# def detecter_trajectoires():
#     import re
#     trajectoires = {}
#     for f in os.listdir(CSV_DIR):
#         m = re.match(r"^gps(?:_(.+))?\.csv$", f)
#         if m:
#             nom  = m.group(1) or "sol"
#             suf  = f"_{nom}" if nom != "sol" else ""
#             gps  = os.path.join(CSV_DIR, f"gps{suf}.csv")
#             imu  = os.path.join(CSV_DIR, f"imu{suf}.csv")
#             odom = os.path.join(CSV_DIR, f"odom{suf}.csv")
#             if all(os.path.isfile(p) for p in [gps, imu, odom]):
#                 trajectoires[nom] = {"gps": gps, "imu": imu, "odom": odom}
#     return trajectoires


# def run_fgo_gps_imu(nom: str, gps_csv: str, imu_csv: str, odom_csv: str):
#     print(f"\n[FGO GPS+IMU] Trajectoire : {nom}")

#     # ── Chargement GPS, conversion metres, recalage sur Odom ───────────────
#     df_gps = pd.read_csv(gps_csv).sort_values("t").reset_index(drop=True)
#     dx, dy, dz = gps_to_meters(df_gps["lat"].to_numpy(), df_gps["lon"].to_numpy(),
#                                 df_gps["alt"].to_numpy())

#     df_odom = pd.read_csv(odom_csv).sort_values("t").reset_index(drop=True)
#     t_od_ref  = df_odom["t"].to_numpy()
#     t_gps_abs = df_gps["t"].to_numpy()
#     x_off = float(np.interp(t_gps_abs[0], t_od_ref, df_odom["x"].to_numpy()))
#     y_off = float(np.interp(t_gps_abs[0], t_od_ref, df_odom["y"].to_numpy()))
#     z_off = float(np.interp(t_gps_abs[0], t_od_ref, df_odom["z"].to_numpy()))
#     dx += x_off; dy += y_off; dz += z_off

#     # ── CORRECTION 1 : debiaisage robuste sur X, Y ET Z ─────────────────────
#     x0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["x"].to_numpy())) for i in range(N_BIAS)])
#     y0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["y"].to_numpy())) for i in range(N_BIAS)])
#     z0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["z"].to_numpy())) for i in range(N_BIAS)])
#     dx -= np.median(dx[:N_BIAS] - x0)
#     dy -= np.median(dy[:N_BIAS] - y0)
#     dz -= np.median(dz[:N_BIAS] - z0)

#     t0 = df_gps["t"].iloc[0]
#     t_gps_full = (df_gps["t"] - t0).to_numpy()

#     # ── CORRECTION 4 : sous-echantillonnage GPS reglable ────────────────────
#     indices = list(range(0, len(dx), DELTA_K))
#     n = len(indices)
#     print(f"[FGO GPS+IMU] {len(dx)} mesures GPS -> {n} noeuds (DELTA_K={DELTA_K})")

#     t_gps = t_gps_full[indices]
#     dx, dy, dz = dx[indices], dy[indices], dz[indices]

#     # ── Chargement IMU ─────────────────────────────────────────────────────
#     df_imu = pd.read_csv(imu_csv).sort_values("t").reset_index(drop=True)
#     t_imu = (df_imu["t"] - t0).to_numpy()
#     ax = df_imu["ax"].to_numpy(); ay = df_imu["ay"].to_numpy(); az = df_imu["az"].to_numpy()

#     mask_climb = t_imu < 7.0
#     print(f"[DEBUG] az montée (t<7s)  : mean={az[mask_climb].mean():.4f} std={az[mask_climb].std():.4f}")
#     print(f"[DEBUG] az reste (t>=7s)  : mean={az[~mask_climb].mean():.4f} std={az[~mask_climb].std():.4f}")


#     if all(c in df_imu.columns for c in ("wx", "wy", "wz")):
#         gx = df_imu["wx"].to_numpy(); gy = df_imu["wy"].to_numpy(); gz = df_imu["wz"].to_numpy()
#     else:
#         print("[FGO GPS+IMU] ATTENTION : colonnes wx/wy/wz absentes -> gyro mis a zero")
#         gx = gy = gz = np.zeros_like(ax)

#     # ── Parametres de preintegration ────────────────────────────────────────
#     params = PreintegrationParams.MakeSharedU(GRAVITY)
#     params.setAccelerometerCovariance(np.eye(3) * ACCEL_SIGMA**2)
#     params.setGyroscopeCovariance(np.eye(3) * GYRO_SIGMA**2)
#     params.setIntegrationCovariance(np.eye(3) * INTEGRATION_COV)

#     # ── Construction du graphe ──────────────────────────────────────────────
#     graph  = gtsam.NonlinearFactorGraph()
#     values = gtsam.Values()

#     cov_gps        = gtsam.noiseModel.Isotropic.Sigma(3, SIGMA_GPS)
#     cov_prior_pose  = gtsam.noiseModel.Diagonal.Sigmas(
#         np.array([SIGMA_PRIOR_ROT]*3 + [SIGMA_PRIOR_POS]*3))
#     cov_prior_vel   = gtsam.noiseModel.Isotropic.Sigma(3, SIGMA_PRIOR_VEL)
#     cov_prior_bias  = gtsam.noiseModel.Isotropic.Sigma(6, SIGMA_PRIOR_BIAS)

#     # ── Noeud 0 : prior (ancre le vol) ──────────────────────────────────────
#     # CORRECTION 5 : mediane des N_BIAS premiers points GPS (deja debiaises)
#     # au lieu du seul premier point brut -- evite d'ancrer tout le graphe
#     # sur un echantillon potentiellement outlier.
#     x0_est = np.median(dx[:N_BIAS])
#     y0_est = np.median(dy[:N_BIAS])
#     z0_est = np.median(dz[:N_BIAS])

#     pose0 = Pose3(Rot3(), Point3(x0_est, y0_est, z0_est))
#     vel0  = np.zeros(3)   # pas de point precedent pour une difference finie
#     bias0 = ConstantBias(np.zeros(3), np.zeros(3))

#     values.insert(X(0), pose0)
#     values.insert(V(0), vel0)
#     values.insert(B(0), bias0)

#     graph.add(gtsam.PriorFactorPose3(X(0), pose0, cov_prior_pose))
#     graph.add(gtsam.PriorFactorVector(V(0), vel0, cov_prior_vel))
#     graph.add(gtsam.PriorFactorConstantBias(B(0), bias0, cov_prior_bias))
#     # CORRECTION 5 : GPSFactor sur X(0) retire -- redondant avec le Prior
#     # maintenant que celui-ci porte une confiance statistiquement coherente
#     # (avant : sigma=0.1 arbitraire dominait de toute facon ce facteur)

#     # ── Noeuds 1..n-1 : GPS + ImuFactor + biais random-walk ────────────────
#     for k in range(1, n):
#         t_prev, t_k = t_gps[k-1], t_gps[k]
#         dt_node = max(t_k - t_prev, 1e-4)

#         pim = PreintegratedImuMeasurements(params, bias0)
#         mask = (t_imu > t_prev) & (t_imu <= t_k)
#         idxs = np.where(mask)[0]
#         t_precedent = t_prev
#         for idx in idxs:
#             dt = max(t_imu[idx] - t_precedent, 1e-4)
#             acc  = np.array([ax[idx], ay[idx], az[idx]])
#             gyro = np.array([gx[idx], gy[idx], gz[idx]])
#             pim.integrateMeasurement(acc, gyro, dt)
#             t_precedent = t_imu[idx]

#         if k >= 2:
#             vel_init = np.array([
#                 dx[k] - dx[k-2], dy[k] - dy[k-2], dz[k] - dz[k-2]
#             ]) / max(t_gps[k] - t_gps[k-2], 1e-3)
#         elif k == 1:
#             vel_init = np.array([dx[k]-dx[k-1], dy[k]-dy[k-1], dz[k]-dz[k-1]]) / dt_node
#         else:
#             vel_init = np.zeros(3)

#         pose_init = Pose3(Rot3(), Point3(dx[k], dy[k], dz[k]))
#         values.insert(X(k), pose_init)
#         values.insert(V(k), vel_init)
#         values.insert(B(k), bias0)

#         graph.add(gtsam.ImuFactor(X(k-1), V(k-1), X(k), V(k), B(k-1), pim))

#         sigma_bias_dt = BIAS_RW_SIGMA * np.sqrt(dt_node)
#         cov_bias_rw = gtsam.noiseModel.Isotropic.Sigma(6, sigma_bias_dt)
#         graph.add(gtsam.BetweenFactorConstantBias(
#             B(k-1), B(k), ConstantBias(np.zeros(3), np.zeros(3)), cov_bias_rw))

#         graph.add(gtsam.GPSFactor(X(k), Point3(dx[k], dy[k], dz[k]), cov_gps))

#     print(f"[FGO GPS+IMU] Graphe : {graph.size()} facteurs, {n} noeuds (Pose+Vel+Bias)")

#     lm_params = gtsam.LevenbergMarquardtParams()
#     lm_params.setMaxIterations(200) #200
#     lm_params.setVerbosity("SILENT")
#     optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params)
#     t_opt_debut = time.time()
#     result = optimizer.optimize()
#     t_opt_fin = time.time()
#     duree_opt = t_opt_fin - t_opt_debut
#     print(f"[FGO GPS+IMU] Temps de calcul (optimisation) : {duree_opt:.4f} s "
#           f"({n} noeuds, {graph.size()} facteurs)")

#     print(f"[FGO GPS+IMU] Erreur initiale : {graph.error(values):.4f}")
#     print(f"[FGO GPS+IMU] Erreur finale   : {graph.error(result):.4f}")

#     x_fgo = np.array([result.atPose3(X(k)).translation()[0] for k in range(n)])
#     y_fgo = np.array([result.atPose3(X(k)).translation()[1] for k in range(n)])
#     z_fgo = np.array([result.atPose3(X(k)).translation()[2] for k in range(n)])

#     biais_finaux = [result.atConstantBias(B(k)) for k in range(n)]
#     biais_acc_final = biais_finaux[-1].accelerometer()
#     print(f"[FGO GPS+IMU] Biais accelerometre estime (final) : {biais_acc_final}")

#     t_odom = (df_odom["t"] - t0).to_numpy()
#     x_odom = np.interp(t_gps, t_odom, df_odom["x"].to_numpy())
#     y_odom = np.interp(t_gps, t_odom, df_odom["y"].to_numpy())
#     z_odom = np.interp(t_gps, t_odom, df_odom["z"].to_numpy())

#     rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
#     rmse_gps_x, rmse_gps_y, rmse_gps_z = rmse(dx, x_odom), rmse(dy, y_odom), rmse(dz, z_odom)
#     rmse_fgo_x, rmse_fgo_y, rmse_fgo_z = rmse(x_fgo, x_odom), rmse(y_fgo, y_odom), rmse(z_fgo, z_odom)

#     print(f"[FGO GPS+IMU] RMSE GPS brut : X={rmse_gps_x:.4f}  Y={rmse_gps_y:.4f}  Z={rmse_gps_z:.4f}")
#     print(f"[FGO GPS+IMU] RMSE FGO      : X={rmse_fgo_x:.4f}  Y={rmse_fgo_y:.4f}  Z={rmse_fgo_z:.4f}")

#     fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
#     fig.suptitle(f"FGO GPS+IMU couplé — {nom}  (temps calcul optimisation : {duree_opt:.3f}s)",
#                  fontsize=11, fontweight="bold")

#     donnees = [
#         ("X — Est/Ouest", dx, x_fgo, x_odom, rmse_gps_x, rmse_fgo_x, "tab:blue"),
#         ("Y — Nord/Sud",  dy, y_fgo, y_odom, rmse_gps_y, rmse_fgo_y, "tab:orange"),
#         ("Z — Altitude",  dz, z_fgo, z_odom, rmse_gps_z, rmse_fgo_z, "tab:green"),
#     ]
#     for ax, (label, brut, fgo, verite, r_gps, r_fgo, color) in zip(axes, donnees):
#         ax.plot(t_gps, verite, "--", color="black", linewidth=1.3, label=f"{label} vérité (Odom)")
#         ax.plot(t_gps, brut, "-", color=color, alpha=0.5, linewidth=1.2, label=f"{label} GPS brut")
#         ax.plot(t_gps, fgo, "-", color="tab:red", linewidth=1.8, label=f"{label} FGO GPS+IMU")
#         ax.text(0.01, 0.95, f"RMSE GPS = {r_gps:.2e} m\nRMSE FGO = {r_fgo:.2e} m",
#                 transform=ax.transAxes, va="top",
#                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
#         ax.set_ylabel(f"{label} (m)")
#         ax.legend(loc="upper right", fontsize=8)
#         ax.grid(True)

#     axes[-1].set_xlabel("Temps (s)")
#     plt.tight_layout()
#     os.makedirs(IMG_DIR, exist_ok=True)
#     out_img = os.path.join(IMG_DIR, f"fgo_imu_gps_coupled_{nom}.pdf")
#     plt.savefig(out_img)
#     print(f"[FGO GPS+IMU] Figure sauvegardée : {out_img}")

#     # ── AJOUT : trace des 6 composantes du biais IMU au fil du vol ─────────
#     # (demande du prof : verifier que les biais convergent, pas juste
#     # regarder la valeur finale)
#     bias_acc = np.array([biais_finaux[k].accelerometer() for k in range(n)])
#     bias_gyro = np.array([biais_finaux[k].gyroscope() for k in range(n)])

#     fig_bias, axes_bias = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
#     fig_bias.suptitle(f"Convergence du biais IMU estime — {nom}", fontsize=11, fontweight="bold")
#     labels_acc = ["bias_ax", "bias_ay", "bias_az"]
#     labels_gyro = ["bias_gx", "bias_gy", "bias_gz"]
#     for i in range(3):
#         axes_bias[i, 0].plot(t_gps, bias_acc[:, i], color="tab:red")
#         axes_bias[i, 0].set_ylabel(f"{labels_acc[i]} (m/s²)")
#         axes_bias[i, 0].grid(True)
#         axes_bias[i, 1].plot(t_gps, bias_gyro[:, i], color="tab:purple")
#         axes_bias[i, 1].set_ylabel(f"{labels_gyro[i]} (rad/s)")
#         axes_bias[i, 1].grid(True)
#     axes_bias[0, 0].set_title("Accelerometer bias")
#     axes_bias[0, 1].set_title("Gyroscope bias")
#     axes_bias[-1, 0].set_xlabel("Temps (s)")
#     axes_bias[-1, 1].set_xlabel("Temps (s)")
#     plt.tight_layout()
#     out_img_bias = os.path.join(IMG_DIR, f"fgo_imu_gps_coupled_{nom}_bias.pdf")
#     plt.savefig(out_img_bias)
#     print(f"[FGO GPS+IMU] Figure biais sauvegardée : {out_img_bias}")

#     # ── AJOUT : trajectoire vue du dessus (X vs Y) ──────────────────────────
#     fig_top, ax_top = plt.subplots(figsize=(8, 8))
#     ax_top.plot(x_odom, y_odom, "--", color="black", linewidth=1.3, label="Vérité (Odom)")
#     ax_top.plot(dx, dy, "-", color="tab:blue", alpha=0.4, linewidth=1.0, label="GPS brut")
#     ax_top.plot(x_fgo, y_fgo, "-", color="tab:red", linewidth=1.8, label="FGO GPS+IMU")
#     ax_top.set_xlabel("X — Est/Ouest (m)")
#     ax_top.set_ylabel("Y — Nord/Sud (m)")
#     ax_top.set_title(f"Trajectoire vue du dessus — {nom}", fontweight="bold")
#     ax_top.legend(fontsize=9)
#     ax_top.grid(True)
#     ax_top.set_aspect("equal", adjustable="box")
#     plt.tight_layout()
#     out_img_top = os.path.join(IMG_DIR, f"fgo_imu_gps_coupled_{nom}_topdown.pdf")
#     plt.savefig(out_img_top)
#     print(f"[FGO GPS+IMU] Figure vue du dessus sauvegardée : {out_img_top}")

#     os.makedirs(OUT_DIR, exist_ok=True)
#     out_csv = os.path.join(OUT_DIR, f"fgo_imu_gps_coupled_{nom}.csv")
#     with open(out_csv, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "z"])
#         writer.writeheader()
#         for k in range(n):
#             writer.writerow({
#                 "t": round(float(t_gps[k]), 4),
#                 "x": round(float(x_fgo[k]), 6),
#                 "y": round(float(y_fgo[k]), 6),
#                 "z": round(float(z_fgo[k]), 6),
#             })
#     print(f"[FGO GPS+IMU] CSV sauvegardé : {out_csv} ({n} lignes)")


# def main():
#     trajectoires = detecter_trajectoires()
#     if not trajectoires:
#         print("[Erreur] Aucun groupe GPS/IMU/Odom trouvé dans csv_files/")
#         return

#     if len(sys.argv) > 1:
#         nom = sys.argv[1]
#         if nom not in trajectoires:
#             print(f"[Erreur] Trajectoire '{nom}' introuvable. "
#                   f"Disponibles : {list(trajectoires.keys())}")
#             return
#         trajectoires = {nom: trajectoires[nom]}

#     for nom, f in trajectoires.items():
#         run_fgo_gps_imu(nom, f["gps"], f["imu"], f["odom"])

#     plt.show()


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
#!/usr/bin/env python3






# """
# fgo_IMU_GPS_coupled.py
# --------------
# FGO couple GPS+IMU en mode batch : un seul graphe de facteurs GTSAM,
# avec preintegration IMU comme facteur (ImuFactor) reliant deux noeuds
# consecutifs, facteur GPS (GPSFactor) sur chaque noeud ou une mesure GPS
# existe, et le biais IMU comme variable estimee (marche aleatoire lente).

# Architecture (cf. papier Cioaca et al., Fig. 2 -- "loosely-coupled
# GNSS/IMU factor graph") :
#   - Noeuds : Pose3 (position+orientation), vitesse, biais, un triplet
#     par instant GPS recu.
#   - Facteur Prior : ancre le premier noeud (pose, vitesse, biais).
#   - Facteur GPS : contrainte absolue sur la position (ignore
#     l'orientation), a chaque instant ou une mesure GPS existe.
#   - Facteur ImuFactor : relie 2 noeuds consecutifs via la preintegration
#     de TOUS les echantillons IMU entre les deux -- y compris pendant
#     une coupure GPS, ou cette fenetre devient simplement plus longue,
#     sans casser le graphe.
#   - Facteur Between sur le biais : modelise sa derive lente (marche
#     aleatoire), permettant a l'optimiseur de le corriger au fil du vol.
#     La covariance de ce facteur est mise a l'echelle de sqrt(dt) --
#     coherent avec un modele de bruit blanc integre (marche aleatoire
#     continue) : Var(b_k - b_{k-1}) = q * dt.

# Contrairement a l'estimateur "uncoupled", il n'y a ICI qu'UN SEUL
# probleme d'optimisation global sur tout le vol : GPS et IMU contribuent
# ensemble au meme cout, et se corrigent mutuellement.

# CORRECTIONS APPLIQUEES (par rapport a la version precedente) :
#   1. Correction de biais sur Z (median robuste sur les N_BIAS premieres
#      mesures), en plus de X/Y deja corriges -- evite l'ancrage sur un
#      premier point GPS isole potentiellement bruite.
#   2. Covariance du biais random-walk mise a l'echelle de sqrt(dt) a
#      chaque iteration, au lieu d'une constante fixe.
#   3. Vitesse initiale par difference finie sur les positions GPS DEJA
#      DEBIAISEES (pas l'Odom, qui ne serait jamais disponible en vrai
#      scenario de vol) -- meilleur point de depart pour l'optimiseur
#      non-lineaire que vel_init=0 partout.
#   4. DELTA_K rendu reglable (defaut=1, comportement inchange) pour
#      permettre de tester l'effet du sous-echantillonnage sur le RMSE.
#   5. Noeud 0 ancre sur la mediane des N_BIAS premiers points GPS
#      (pas le premier point brut isole) + SIGMA_PRIOR_POS coherent
#      statistiquement avec cet estimateur + GPSFactor redondant sur
#      X(0) retire.
#   6. AJOUT : trace des 6 composantes du biais IMU estime (accel x,y,z
#      + gyro x,y,z) au fil du vol -- demande du prof, pour verifier
#      visuellement qu'elles convergent.
#   7. AJOUT (cette session) : temps de calcul optimisation affiche aussi
#      dans le titre de la figure "vue du dessus" (topdown), en plus de
#      la figure principale et du terminal.

# Usage :
#   python3 fusion/fgo_IMU_GPS_coupled.py vertical
#   python3 fusion/fgo_IMU_GPS_coupled.py               <- detecte toutes les trajectoires
# """

# import sys, os, time
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# import csv
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import gtsam
# from gtsam import symbol_shorthand
# from gtsam import PreintegrationParams, PreintegratedImuMeasurements
# from gtsam import Pose3, Rot3, Point3
# from gtsam.imuBias import ConstantBias

# X = symbol_shorthand.X   # Pose3 (position + orientation)
# V = symbol_shorthand.V   # Vector3 (vitesse)
# B = symbol_shorthand.B   # ConstantBias (biais accel+gyro)

# _ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# CSV_DIR = os.path.join(_ROOT, "csv_files")
# OUT_DIR = os.path.join(CSV_DIR, "fgo_imu_gps_coupled_csv")
# IMG_DIR = os.path.join(_ROOT, "images")

# GRAVITY = 9.8

# # ─── Parametres modifiables ───────────────────────────────────────────────
# SIGMA_GPS        = 1.0     # m       -- coherent avec le bruit SDF homogene
# N_BIAS           = 15      # nb de mesures GPS utilisees pour le debiaisage initial
# SIGMA_PRIOR_POS  = SIGMA_GPS / np.sqrt(N_BIAS)  # ~0.258 m -- coherent avec la
#                            # precision statistique d'une mediane sur N_BIAS points
#                            # (au lieu d'une valeur fixe arbitraire 10x trop confiante)
# SIGMA_PRIOR_ROT  = 0.1     # rad
# SIGMA_PRIOR_VEL  = 0.1     # m/s
# SIGMA_PRIOR_BIAS = 0.1     # -- incertitude initiale sur le biais

# ACCEL_SIGMA      = 0.173    # m/s^2   -- bruit reel du SDF (accelerometre)
# GYRO_SIGMA       = 0.02    # rad/s   -- bruit reel du SDF (gyroscope)
# INTEGRATION_COV  = 1e-8
# BIAS_RW_SIGMA    = 5e-3    # 5e-3 taux de derive du biais (sigma "par racine de seconde")
#                            # -- teste x10 : 5e-4 figeait le biais sur vol "carre" (62s)
#                            # -- la covariance reelle utilisee = BIAS_RW_SIGMA * sqrt(dt)

# DELTA_K          = 1       # sous-echantillonnage GPS : 1 noeud tous les DELTA_K points
#                            # (1 = comportement d'origine, aucun sous-echantillonnage)
# # ────────────────────────────────────────────────────────────────────────


# def gps_to_meters(lat, lon, alt):
#     """Identique aux scripts precedents : conversion lat/lon/alt -> metres."""
#     R     = 6_371_000.0
#     N_REF = min(10, len(lat))
#     lat_rad = np.radians(lat)
#     lat0  = np.median(lat_rad[:N_REF])
#     lon0  = np.median(lon[:N_REF])
#     alt0  = np.median(alt[:N_REF])
#     dx = (lon - lon0) * np.cos(lat0) * np.radians(R)
#     dy = (lat - np.median(lat[:N_REF])) * np.radians(R)
#     dz = alt - alt0
#     return dx, dy, dz


# def detecter_trajectoires():
#     import re
#     trajectoires = {}
#     for f in os.listdir(CSV_DIR):
#         m = re.match(r"^gps(?:_(.+))?\.csv$", f)
#         if m:
#             nom  = m.group(1) or "sol"
#             suf  = f"_{nom}" if nom != "sol" else ""
#             gps  = os.path.join(CSV_DIR, f"gps{suf}.csv")
#             imu  = os.path.join(CSV_DIR, f"imu{suf}.csv")
#             odom = os.path.join(CSV_DIR, f"odom{suf}.csv")
#             if all(os.path.isfile(p) for p in [gps, imu, odom]):
#                 trajectoires[nom] = {"gps": gps, "imu": imu, "odom": odom}
#     return trajectoires


# def run_fgo_gps_imu(nom: str, gps_csv: str, imu_csv: str, odom_csv: str):
#     print(f"\n[FGO GPS+IMU] Trajectoire : {nom}")

#     # ── Chargement GPS, conversion metres, recalage sur Odom ───────────────
#     df_gps = pd.read_csv(gps_csv).sort_values("t").reset_index(drop=True)
#     dx, dy, dz = gps_to_meters(df_gps["lat"].to_numpy(), df_gps["lon"].to_numpy(),
#                                 df_gps["alt"].to_numpy())

#     df_odom = pd.read_csv(odom_csv).sort_values("t").reset_index(drop=True)
#     t_od_ref  = df_odom["t"].to_numpy()
#     t_gps_abs = df_gps["t"].to_numpy()
#     x_off = float(np.interp(t_gps_abs[0], t_od_ref, df_odom["x"].to_numpy()))
#     y_off = float(np.interp(t_gps_abs[0], t_od_ref, df_odom["y"].to_numpy()))
#     z_off = float(np.interp(t_gps_abs[0], t_od_ref, df_odom["z"].to_numpy()))
#     dx += x_off; dy += y_off; dz += z_off

#     # ── CORRECTION 1 : debiaisage robuste sur X, Y ET Z ─────────────────────
#     x0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["x"].to_numpy())) for i in range(N_BIAS)])
#     y0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["y"].to_numpy())) for i in range(N_BIAS)])
#     z0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["z"].to_numpy())) for i in range(N_BIAS)])
#     dx -= np.median(dx[:N_BIAS] - x0)
#     dy -= np.median(dy[:N_BIAS] - y0)
#     dz -= np.median(dz[:N_BIAS] - z0)

#     t0 = df_gps["t"].iloc[0]
#     t_gps_full = (df_gps["t"] - t0).to_numpy()

#     # ── CORRECTION 4 : sous-echantillonnage GPS reglable ────────────────────
#     indices = list(range(0, len(dx), DELTA_K))
#     n = len(indices)
#     print(f"[FGO GPS+IMU] {len(dx)} mesures GPS -> {n} noeuds (DELTA_K={DELTA_K})")

#     t_gps = t_gps_full[indices]
#     dx, dy, dz = dx[indices], dy[indices], dz[indices]

#     # ── Chargement IMU ─────────────────────────────────────────────────────
#     df_imu = pd.read_csv(imu_csv).sort_values("t").reset_index(drop=True)
#     t_imu = (df_imu["t"] - t0).to_numpy()
#     ax = df_imu["ax"].to_numpy(); ay = df_imu["ay"].to_numpy(); az = df_imu["az"].to_numpy()

#     mask_climb = t_imu < 7.0
#     print(f"[DEBUG] az montée (t<7s)  : mean={az[mask_climb].mean():.4f} std={az[mask_climb].std():.4f}")
#     print(f"[DEBUG] az reste (t>=7s)  : mean={az[~mask_climb].mean():.4f} std={az[~mask_climb].std():.4f}")


#     if all(c in df_imu.columns for c in ("wx", "wy", "wz")):
#         gx = df_imu["wx"].to_numpy(); gy = df_imu["wy"].to_numpy(); gz = df_imu["wz"].to_numpy()
#     else:
#         print("[FGO GPS+IMU] ATTENTION : colonnes wx/wy/wz absentes -> gyro mis a zero")
#         gx = gy = gz = np.zeros_like(ax)

#     # ── Parametres de preintegration ────────────────────────────────────────
#     params = PreintegrationParams.MakeSharedU(GRAVITY)
#     params.setAccelerometerCovariance(np.eye(3) * ACCEL_SIGMA**2)
#     params.setGyroscopeCovariance(np.eye(3) * GYRO_SIGMA**2)
#     params.setIntegrationCovariance(np.eye(3) * INTEGRATION_COV)

#     # ── Construction du graphe ──────────────────────────────────────────────
#     graph  = gtsam.NonlinearFactorGraph()
#     values = gtsam.Values()

#     cov_gps        = gtsam.noiseModel.Isotropic.Sigma(3, SIGMA_GPS)
#     cov_prior_pose  = gtsam.noiseModel.Diagonal.Sigmas(
#         np.array([SIGMA_PRIOR_ROT]*3 + [SIGMA_PRIOR_POS]*3))
#     cov_prior_vel   = gtsam.noiseModel.Isotropic.Sigma(3, SIGMA_PRIOR_VEL)
#     cov_prior_bias  = gtsam.noiseModel.Isotropic.Sigma(6, SIGMA_PRIOR_BIAS)

#     # ── Noeud 0 : prior (ancre le vol) ──────────────────────────────────────
#     # CORRECTION 5 : mediane des N_BIAS premiers points GPS (deja debiaises)
#     # au lieu du seul premier point brut -- evite d'ancrer tout le graphe
#     # sur un echantillon potentiellement outlier.
#     x0_est = np.median(dx[:N_BIAS])
#     y0_est = np.median(dy[:N_BIAS])
#     z0_est = np.median(dz[:N_BIAS])

#     pose0 = Pose3(Rot3(), Point3(x0_est, y0_est, z0_est))
#     vel0  = np.zeros(3)   # pas de point precedent pour une difference finie
#     bias0 = ConstantBias(np.zeros(3), np.zeros(3))

#     values.insert(X(0), pose0)
#     values.insert(V(0), vel0)
#     values.insert(B(0), bias0)

#     graph.add(gtsam.PriorFactorPose3(X(0), pose0, cov_prior_pose))
#     graph.add(gtsam.PriorFactorVector(V(0), vel0, cov_prior_vel))
#     graph.add(gtsam.PriorFactorConstantBias(B(0), bias0, cov_prior_bias))
#     # CORRECTION 5 : GPSFactor sur X(0) retire -- redondant avec le Prior
#     # maintenant que celui-ci porte une confiance statistiquement coherente
#     # (avant : sigma=0.1 arbitraire dominait de toute facon ce facteur)

#     # ── Noeuds 1..n-1 : GPS + ImuFactor + biais random-walk ────────────────
#     for k in range(1, n):
#         t_prev, t_k = t_gps[k-1], t_gps[k]
#         dt_node = max(t_k - t_prev, 1e-4)

#         pim = PreintegratedImuMeasurements(params, bias0)
#         mask = (t_imu > t_prev) & (t_imu <= t_k)
#         idxs = np.where(mask)[0]
#         t_precedent = t_prev
#         for idx in idxs:
#             dt = max(t_imu[idx] - t_precedent, 1e-4)
#             acc  = np.array([ax[idx], ay[idx], az[idx]])
#             gyro = np.array([gx[idx], gy[idx], gz[idx]])
#             pim.integrateMeasurement(acc, gyro, dt)
#             t_precedent = t_imu[idx]

#         if k >= 2:
#             vel_init = np.array([
#                 dx[k] - dx[k-2], dy[k] - dy[k-2], dz[k] - dz[k-2]
#             ]) / max(t_gps[k] - t_gps[k-2], 1e-3)
#         elif k == 1:
#             vel_init = np.array([dx[k]-dx[k-1], dy[k]-dy[k-1], dz[k]-dz[k-1]]) / dt_node
#         else:
#             vel_init = np.zeros(3)

#         pose_init = Pose3(Rot3(), Point3(dx[k], dy[k], dz[k]))
#         values.insert(X(k), pose_init)
#         values.insert(V(k), vel_init)
#         values.insert(B(k), bias0)

#         graph.add(gtsam.ImuFactor(X(k-1), V(k-1), X(k), V(k), B(k-1), pim))

#         sigma_bias_dt = BIAS_RW_SIGMA * np.sqrt(dt_node)
#         cov_bias_rw = gtsam.noiseModel.Isotropic.Sigma(6, sigma_bias_dt)
#         graph.add(gtsam.BetweenFactorConstantBias(
#             B(k-1), B(k), ConstantBias(np.zeros(3), np.zeros(3)), cov_bias_rw))

#         graph.add(gtsam.GPSFactor(X(k), Point3(dx[k], dy[k], dz[k]), cov_gps))

#     print(f"[FGO GPS+IMU] Graphe : {graph.size()} facteurs, {n} noeuds (Pose+Vel+Bias)")

#     lm_params = gtsam.LevenbergMarquardtParams()
#     lm_params.setMaxIterations(200) #200
#     lm_params.setVerbosity("SILENT")
#     optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params)
#     t_opt_debut = time.time()
#     result = optimizer.optimize()
#     t_opt_fin = time.time()
#     duree_opt = t_opt_fin - t_opt_debut
#     print(f"[FGO GPS+IMU] Temps de calcul (optimisation) : {duree_opt:.4f} s "
#           f"({n} noeuds, {graph.size()} facteurs)")

#     print(f"[FGO GPS+IMU] Erreur initiale : {graph.error(values):.4f}")
#     print(f"[FGO GPS+IMU] Erreur finale   : {graph.error(result):.4f}")

#     x_fgo = np.array([result.atPose3(X(k)).translation()[0] for k in range(n)])
#     y_fgo = np.array([result.atPose3(X(k)).translation()[1] for k in range(n)])
#     z_fgo = np.array([result.atPose3(X(k)).translation()[2] for k in range(n)])

#     biais_finaux = [result.atConstantBias(B(k)) for k in range(n)]
#     biais_acc_final = biais_finaux[-1].accelerometer()
#     print(f"[FGO GPS+IMU] Biais accelerometre estime (final) : {biais_acc_final}")

#     t_odom = (df_odom["t"] - t0).to_numpy()
#     x_odom = np.interp(t_gps, t_odom, df_odom["x"].to_numpy())
#     y_odom = np.interp(t_gps, t_odom, df_odom["y"].to_numpy())
#     z_odom = np.interp(t_gps, t_odom, df_odom["z"].to_numpy())

#     rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
#     rmse_gps_x, rmse_gps_y, rmse_gps_z = rmse(dx, x_odom), rmse(dy, y_odom), rmse(dz, z_odom)
#     rmse_fgo_x, rmse_fgo_y, rmse_fgo_z = rmse(x_fgo, x_odom), rmse(y_fgo, y_odom), rmse(z_fgo, z_odom)

#     print(f"[FGO GPS+IMU] RMSE GPS brut : X={rmse_gps_x:.4f}  Y={rmse_gps_y:.4f}  Z={rmse_gps_z:.4f}")
#     print(f"[FGO GPS+IMU] RMSE FGO      : X={rmse_fgo_x:.4f}  Y={rmse_fgo_y:.4f}  Z={rmse_fgo_z:.4f}")

#     fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
#     fig.suptitle(f"FGO GPS+IMU couplé — {nom}  (temps calcul optimisation : {duree_opt:.3f}s)",
#                  fontsize=11, fontweight="bold")

#     donnees = [
#         ("X — Est/Ouest", dx, x_fgo, x_odom, rmse_gps_x, rmse_fgo_x, "tab:blue"),
#         ("Y — Nord/Sud",  dy, y_fgo, y_odom, rmse_gps_y, rmse_fgo_y, "tab:orange"),
#         ("Z — Altitude",  dz, z_fgo, z_odom, rmse_gps_z, rmse_fgo_z, "tab:green"),
#     ]
#     for ax, (label, brut, fgo, verite, r_gps, r_fgo, color) in zip(axes, donnees):
#         ax.plot(t_gps, verite, "--", color="black", linewidth=1.3, label=f"{label} vérité (Odom)")
#         ax.plot(t_gps, brut, "-", color=color, alpha=0.5, linewidth=1.2, label=f"{label} GPS brut")
#         ax.plot(t_gps, fgo, "-", color="tab:red", linewidth=1.8, label=f"{label} FGO GPS+IMU")
#         ax.text(0.01, 0.95, f"RMSE GPS = {r_gps:.2e} m\nRMSE FGO = {r_fgo:.2e} m",
#                 transform=ax.transAxes, va="top",
#                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
#         ax.set_ylabel(f"{label} (m)")
#         ax.legend(loc="upper right", fontsize=8)
#         ax.grid(True)

#     axes[-1].set_xlabel("Temps (s)")
#     plt.tight_layout()
#     os.makedirs(IMG_DIR, exist_ok=True)
#     out_img = os.path.join(IMG_DIR, f"fgo_imu_gps_coupled_{nom}.pdf")
#     plt.savefig(out_img)
#     print(f"[FGO GPS+IMU] Figure sauvegardée : {out_img}")

#     # ── AJOUT : trace des 6 composantes du biais IMU au fil du vol ─────────
#     # (demande du prof : verifier que les biais convergent, pas juste
#     # regarder la valeur finale)
#     bias_acc = np.array([biais_finaux[k].accelerometer() for k in range(n)])
#     bias_gyro = np.array([biais_finaux[k].gyroscope() for k in range(n)])

#     fig_bias, axes_bias = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
#     fig_bias.suptitle(f"Convergence du biais IMU estime — {nom}", fontsize=11, fontweight="bold")
#     labels_acc = ["bias_ax", "bias_ay", "bias_az"]
#     labels_gyro = ["bias_gx", "bias_gy", "bias_gz"]
#     for i in range(3):
#         axes_bias[i, 0].plot(t_gps, bias_acc[:, i], color="tab:red")
#         axes_bias[i, 0].set_ylabel(f"{labels_acc[i]} (m/s²)")
#         axes_bias[i, 0].grid(True)
#         axes_bias[i, 1].plot(t_gps, bias_gyro[:, i], color="tab:purple")
#         axes_bias[i, 1].set_ylabel(f"{labels_gyro[i]} (rad/s)")
#         axes_bias[i, 1].grid(True)
#     axes_bias[0, 0].set_title("Accelerometer bias")
#     axes_bias[0, 1].set_title("Gyroscope bias")
#     axes_bias[-1, 0].set_xlabel("Temps (s)")
#     axes_bias[-1, 1].set_xlabel("Temps (s)")
#     plt.tight_layout()
#     out_img_bias = os.path.join(IMG_DIR, f"fgo_imu_gps_coupled_{nom}_bias.pdf")
#     plt.savefig(out_img_bias)
#     print(f"[FGO GPS+IMU] Figure biais sauvegardée : {out_img_bias}")

#     # ── AJOUT : trajectoire vue du dessus (X vs Y) ──────────────────────────
#     # (cette session : ajout du temps de calcul dans le titre, comme demande)
#     fig_top, ax_top = plt.subplots(figsize=(8, 8))
#     ax_top.plot(x_odom, y_odom, "--", color="black", linewidth=1.3, label="Vérité (Odom)")
#     ax_top.plot(dx, dy, "-", color="tab:blue", alpha=0.4, linewidth=1.0, label="GPS brut")
#     ax_top.plot(x_fgo, y_fgo, "-", color="tab:red", linewidth=1.8, label="FGO GPS+IMU")
#     ax_top.set_xlabel("X — Est/Ouest (m)")
#     ax_top.set_ylabel("Y — Nord/Sud (m)")
#     ax_top.set_title(f"Trajectoire vue du dessus — {nom}  (temps calcul optimisation : {duree_opt:.3f}s)",
#                       fontweight="bold")
#     ax_top.legend(fontsize=9)
#     ax_top.grid(True)
#     ax_top.set_aspect("equal", adjustable="box")
#     plt.tight_layout()
#     out_img_top = os.path.join(IMG_DIR, f"fgo_imu_gps_coupled_{nom}_topdown.pdf")
#     plt.savefig(out_img_top)
#     print(f"[FGO GPS+IMU] Figure vue du dessus sauvegardée : {out_img_top}")

#     os.makedirs(OUT_DIR, exist_ok=True)
#     out_csv = os.path.join(OUT_DIR, f"fgo_imu_gps_coupled_{nom}.csv")
#     with open(out_csv, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "z"])
#         writer.writeheader()
#         for k in range(n):
#             writer.writerow({
#                 "t": round(float(t_gps[k]), 4),
#                 "x": round(float(x_fgo[k]), 6),
#                 "y": round(float(y_fgo[k]), 6),
#                 "z": round(float(z_fgo[k]), 6),
#             })
#     print(f"[FGO GPS+IMU] CSV sauvegardé : {out_csv} ({n} lignes)")


# def main():
#     trajectoires = detecter_trajectoires()
#     if not trajectoires:
#         print("[Erreur] Aucun groupe GPS/IMU/Odom trouvé dans csv_files/")
#         return

#     if len(sys.argv) > 1:
#         nom = sys.argv[1]
#         if nom not in trajectoires:
#             print(f"[Erreur] Trajectoire '{nom}' introuvable. "
#                   f"Disponibles : {list(trajectoires.keys())}")
#             return
#         trajectoires = {nom: trajectoires[nom]}

#     for nom, f in trajectoires.items():
#         run_fgo_gps_imu(nom, f["gps"], f["imu"], f["odom"])

#     plt.show()


# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
"""
fgo_IMU_GPS_coupled.py
--------------
FGO couple GPS+IMU en mode batch : un seul graphe de facteurs GTSAM,
avec preintegration IMU comme facteur (ImuFactor) reliant deux noeuds
consecutifs, facteur GPS (GPSFactor) sur chaque noeud ou une mesure GPS
existe, et le biais IMU comme variable estimee (marche aleatoire lente).

Architecture (cf. papier Cioaca et al., Fig. 2 -- "loosely-coupled
GNSS/IMU factor graph") :
  - Noeuds : Pose3 (position+orientation), vitesse, biais, un triplet
    par instant GPS recu.
  - Facteur Prior : ancre le premier noeud (pose, vitesse, biais).
  - Facteur GPS : contrainte absolue sur la position (ignore
    l'orientation), a chaque instant ou une mesure GPS existe.
  - Facteur ImuFactor : relie 2 noeuds consecutifs via la preintegration
    de TOUS les echantillons IMU entre les deux -- y compris pendant
    une coupure GPS, ou cette fenetre devient simplement plus longue,
    sans casser le graphe.
  - Facteur Between sur le biais : modelise sa derive lente (marche
    aleatoire), permettant a l'optimiseur de le corriger au fil du vol.
    La covariance de ce facteur est mise a l'echelle de sqrt(dt) --
    coherent avec un modele de bruit blanc integre (marche aleatoire
    continue) : Var(b_k - b_{k-1}) = q * dt.

Contrairement a l'estimateur "uncoupled", il n'y a ICI qu'UN SEUL
probleme d'optimisation global sur tout le vol : GPS et IMU contribuent
ensemble au meme cout, et se corrigent mutuellement.

CORRECTIONS APPLIQUEES (par rapport a la version precedente) :
  1. Correction de biais sur Z (median robuste sur les N_BIAS premieres
     mesures), en plus de X/Y deja corriges -- evite l'ancrage sur un
     premier point GPS isole potentiellement bruite.
  2. Covariance du biais random-walk mise a l'echelle de sqrt(dt) a
     chaque iteration, au lieu d'une constante fixe.
  3. Vitesse initiale par difference finie sur les positions GPS DEJA
     DEBIAISEES (pas l'Odom, qui ne serait jamais disponible en vrai
     scenario de vol) -- meilleur point de depart pour l'optimiseur
     non-lineaire que vel_init=0 partout.
  4. DELTA_K rendu reglable (defaut=1, comportement inchange) pour
     permettre de tester l'effet du sous-echantillonnage sur le RMSE.
  5. Noeud 0 ancre sur la mediane des N_BIAS premiers points GPS
     (pas le premier point brut isole) + SIGMA_PRIOR_POS coherent
     statistiquement avec cet estimateur + GPSFactor redondant sur
     X(0) retire.
  6. AJOUT : trace des 6 composantes du biais IMU estime (accel x,y,z
     + gyro x,y,z) au fil du vol -- demande du prof, pour verifier
     visuellement qu'elles convergent.
  7. AJOUT (cette session) : temps de calcul optimisation affiche aussi
     dans le titre de la figure "vue du dessus" (topdown), en plus de
     la figure principale et du terminal.

Usage :
  python3 fusion/fgo_IMU_GPS_coupled.py vertical
  python3 fusion/fgo_IMU_GPS_coupled.py               <- detecte toutes les trajectoires
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

X = symbol_shorthand.X   # Pose3 (position + orientation)
V = symbol_shorthand.V   # Vector3 (vitesse)
B = symbol_shorthand.B   # ConstantBias (biais accel+gyro)

_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(_ROOT, "csv_files")
OUT_DIR = os.path.join(CSV_DIR, "fgo_imu_gps_coupled_csv")
IMG_DIR = os.path.join(_ROOT, "images")

GRAVITY = 9.8

# ─── Parametres modifiables ───────────────────────────────────────────────
SIGMA_GPS        = 1.0     # m       -- coherent avec le bruit SDF homogene
N_BIAS           = 15      # nb de mesures GPS utilisees pour le debiaisage initial
SIGMA_PRIOR_POS  = SIGMA_GPS / np.sqrt(N_BIAS)  # ~0.258 m -- coherent avec la
                           # precision statistique d'une mediane sur N_BIAS points
                           # (au lieu d'une valeur fixe arbitraire 10x trop confiante)
SIGMA_PRIOR_ROT  = 0.1     # rad
SIGMA_PRIOR_VEL  = 0.1     # m/s
SIGMA_PRIOR_BIAS = 0.1     # -- incertitude initiale sur le biais

ACCEL_SIGMA      = 0.173    # m/s^2   -- bruit reel du SDF (accelerometre)
GYRO_SIGMA       = 0.02    # rad/s   -- bruit reel du SDF (gyroscope)
INTEGRATION_COV  = 1e-8
BIAS_RW_SIGMA    = 2e-3    # RESSERRE (etait 5e-3) : maintenant que la contrainte de
                           # vitesse GPS (voir plus bas) aide deja a stabiliser le
                           # biais independamment, on peut se permettre de moins le
                           # laisser deriver librement -- reduit le risque qu'il
                           # "s'emballe" et cree des boucles dans les virages.

DELTA_K          = 1       # sous-echantillonnage GPS : 1 noeud tous les DELTA_K points
                           # (1 = comportement d'origine, aucun sous-echantillonnage)
# ────────────────────────────────────────────────────────────────────────


def gps_to_meters(lat, lon, alt):
    """Identique aux scripts precedents : conversion lat/lon/alt -> metres."""
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
    import re
    trajectoires = {}
    for f in os.listdir(CSV_DIR):
        m = re.match(r"^gps(?:_(.+))?\.csv$", f)
        if m:
            nom  = m.group(1) or "sol"
            suf  = f"_{nom}" if nom != "sol" else ""
            gps  = os.path.join(CSV_DIR, f"gps{suf}.csv")
            imu  = os.path.join(CSV_DIR, f"imu{suf}.csv")
            odom = os.path.join(CSV_DIR, f"odom{suf}.csv")
            if all(os.path.isfile(p) for p in [gps, imu, odom]):
                trajectoires[nom] = {"gps": gps, "imu": imu, "odom": odom}
    return trajectoires


def run_fgo_gps_imu(nom: str, gps_csv: str, imu_csv: str, odom_csv: str):
    print(f"\n[FGO GPS+IMU] Trajectoire : {nom}")

    # ── Chargement GPS, conversion metres, recalage sur Odom ───────────────
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

    # ── CORRECTION 1 : debiaisage robuste sur X, Y ET Z ─────────────────────
    x0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["x"].to_numpy())) for i in range(N_BIAS)])
    y0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["y"].to_numpy())) for i in range(N_BIAS)])
    z0 = np.array([float(np.interp(t_gps_abs[i], t_od_ref, df_odom["z"].to_numpy())) for i in range(N_BIAS)])
    dx -= np.median(dx[:N_BIAS] - x0)
    dy -= np.median(dy[:N_BIAS] - y0)
    dz -= np.median(dz[:N_BIAS] - z0)

    t0 = df_gps["t"].iloc[0]
    t_gps_full = (df_gps["t"] - t0).to_numpy()

    # ── CORRECTION 4 : sous-echantillonnage GPS reglable ────────────────────
    indices = list(range(0, len(dx), DELTA_K))
    n = len(indices)
    print(f"[FGO GPS+IMU] {len(dx)} mesures GPS -> {n} noeuds (DELTA_K={DELTA_K})")

    t_gps = t_gps_full[indices]
    dx, dy, dz = dx[indices], dy[indices], dz[indices]

    # ── Chargement IMU ─────────────────────────────────────────────────────
    df_imu = pd.read_csv(imu_csv).sort_values("t").reset_index(drop=True)
    t_imu = (df_imu["t"] - t0).to_numpy()
    ax = df_imu["ax"].to_numpy(); ay = df_imu["ay"].to_numpy(); az = df_imu["az"].to_numpy()

    mask_climb = t_imu < 7.0
    print(f"[DEBUG] az montée (t<7s)  : mean={az[mask_climb].mean():.4f} std={az[mask_climb].std():.4f}")
    print(f"[DEBUG] az reste (t>=7s)  : mean={az[~mask_climb].mean():.4f} std={az[~mask_climb].std():.4f}")


    if all(c in df_imu.columns for c in ("wx", "wy", "wz")):
        gx = df_imu["wx"].to_numpy(); gy = df_imu["wy"].to_numpy(); gz = df_imu["wz"].to_numpy()
    else:
        print("[FGO GPS+IMU] ATTENTION : colonnes wx/wy/wz absentes -> gyro mis a zero")
        gx = gy = gz = np.zeros_like(ax)

    # ── Parametres de preintegration ────────────────────────────────────────
    params = PreintegrationParams.MakeSharedU(GRAVITY)
    params.setAccelerometerCovariance(np.eye(3) * ACCEL_SIGMA**2)
    params.setGyroscopeCovariance(np.eye(3) * GYRO_SIGMA**2)
    params.setIntegrationCovariance(np.eye(3) * INTEGRATION_COV)

    # ── Construction du graphe ──────────────────────────────────────────────
    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # cov_gps : bruit gaussien standard, EMBALLE dans une perte robuste de
    # Huber -- au-dela d'un certain ecart residuel, la penalite devient
    # LINEAIRE au lieu de QUADRATIQUE, donc un point GPS aberrant (pic de
    # bruit) "tire" beaucoup moins fort sur la trajectoire qu'avec une
    # gaussienne pure. Technique standard pour rendre l'optimisation
    # resistante aux outliers, sans avoir a les detecter/retirer a la main.
    cov_gps_gaussienne = gtsam.noiseModel.Isotropic.Sigma(3, SIGMA_GPS)
    cov_gps = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), cov_gps_gaussienne)
    cov_prior_pose  = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([SIGMA_PRIOR_ROT]*3 + [SIGMA_PRIOR_POS]*3))
    cov_prior_vel   = gtsam.noiseModel.Isotropic.Sigma(3, SIGMA_PRIOR_VEL)
    cov_prior_bias  = gtsam.noiseModel.Isotropic.Sigma(6, SIGMA_PRIOR_BIAS)

    # ── Noeud 0 : prior (ancre le vol) ──────────────────────────────────────
    # CORRECTION 5 : mediane des N_BIAS premiers points GPS (deja debiaises)
    # au lieu du seul premier point brut -- evite d'ancrer tout le graphe
    # sur un echantillon potentiellement outlier.
    x0_est = np.median(dx[:N_BIAS])
    y0_est = np.median(dy[:N_BIAS])
    z0_est = np.median(dz[:N_BIAS])

    pose0 = Pose3(Rot3(), Point3(x0_est, y0_est, z0_est))
    vel0  = np.zeros(3)   # pas de point precedent pour une difference finie
    bias0 = ConstantBias(np.zeros(3), np.zeros(3))

    values.insert(X(0), pose0)
    values.insert(V(0), vel0)
    values.insert(B(0), bias0)

    graph.add(gtsam.PriorFactorPose3(X(0), pose0, cov_prior_pose))
    graph.add(gtsam.PriorFactorVector(V(0), vel0, cov_prior_vel))
    graph.add(gtsam.PriorFactorConstantBias(B(0), bias0, cov_prior_bias))
    # CORRECTION 5 : GPSFactor sur X(0) retire -- redondant avec le Prior
    # maintenant que celui-ci porte une confiance statistiquement coherente
    # (avant : sigma=0.1 arbitraire dominait de toute facon ce facteur)

    # ── Noeuds 1..n-1 : GPS + ImuFactor + biais random-walk ────────────────
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
            dt_vel = max(t_gps[k] - t_gps[k-2], 1e-3)
            vel_init = np.array([
                dx[k] - dx[k-2], dy[k] - dy[k-2], dz[k] - dz[k-2]
            ]) / dt_vel
        elif k == 1:
            dt_vel = dt_node
            vel_init = np.array([dx[k]-dx[k-1], dy[k]-dy[k-1], dz[k]-dz[k-1]]) / dt_vel
        else:
            dt_vel = dt_node
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

        # ── AJOUT (cette session) : contrainte de VITESSE derivee du GPS ────
        # (difference finie entre 2 points GPS), en plus de la position.
        # C'est une 2eme source d'information INDEPENDANTE de l'IMU, sans
        # materiel supplementaire (pas de Flow Deck) -- technique standard
        # en fusion GPS/INS. Pendant un virage, l'IMU seul confond
        # inclinaison et acceleration horizontale (limite d'observabilite
        # physique, documentee dans la litterature) ; cette contrainte donne
        # a l'optimiseur un second moyen de detecter/corriger la derive du
        # biais, independamment de cette ambiguite.
        # Propagation d'incertitude : v = (p_k - p_{k-2}) / dt, chaque
        # position ayant un bruit independant sigma=SIGMA_GPS, donc
        # sigma_v = SIGMA_GPS * sqrt(2) / dt (somme quadratique de 2 sources
        # independantes, divisee par dt).
        sigma_v = SIGMA_GPS * np.sqrt(2) / dt_vel
        cov_vel_gps = gtsam.noiseModel.Isotropic.Sigma(3, sigma_v)
        graph.add(gtsam.PriorFactorVector(V(k), vel_init, cov_vel_gps))

    print(f"[FGO GPS+IMU] Graphe : {graph.size()} facteurs, {n} noeuds (Pose+Vel+Bias)")

    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(200) #200
    lm_params.setVerbosity("SILENT")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params)
    t_opt_debut = time.time()
    result = optimizer.optimize()
    t_opt_fin = time.time()
    duree_opt = t_opt_fin - t_opt_debut
    print(f"[FGO GPS+IMU] Temps de calcul (optimisation) : {duree_opt:.4f} s "
          f"({n} noeuds, {graph.size()} facteurs)")

    print(f"[FGO GPS+IMU] Erreur initiale : {graph.error(values):.4f}")
    print(f"[FGO GPS+IMU] Erreur finale   : {graph.error(result):.4f}")

    x_fgo = np.array([result.atPose3(X(k)).translation()[0] for k in range(n)])
    y_fgo = np.array([result.atPose3(X(k)).translation()[1] for k in range(n)])
    z_fgo = np.array([result.atPose3(X(k)).translation()[2] for k in range(n)])

    biais_finaux = [result.atConstantBias(B(k)) for k in range(n)]
    biais_acc_final = biais_finaux[-1].accelerometer()
    print(f"[FGO GPS+IMU] Biais accelerometre estime (final) : {biais_acc_final}")

    t_odom = (df_odom["t"] - t0).to_numpy()
    x_odom = np.interp(t_gps, t_odom, df_odom["x"].to_numpy())
    y_odom = np.interp(t_gps, t_odom, df_odom["y"].to_numpy())
    z_odom = np.interp(t_gps, t_odom, df_odom["z"].to_numpy())

    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
    rmse_gps_x, rmse_gps_y, rmse_gps_z = rmse(dx, x_odom), rmse(dy, y_odom), rmse(dz, z_odom)
    rmse_fgo_x, rmse_fgo_y, rmse_fgo_z = rmse(x_fgo, x_odom), rmse(y_fgo, y_odom), rmse(z_fgo, z_odom)

    print(f"[FGO GPS+IMU] RMSE GPS brut : X={rmse_gps_x:.4f}  Y={rmse_gps_y:.4f}  Z={rmse_gps_z:.4f}")
    print(f"[FGO GPS+IMU] RMSE FGO      : X={rmse_fgo_x:.4f}  Y={rmse_fgo_y:.4f}  Z={rmse_fgo_z:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"FGO GPS+IMU couplé — {nom}  (temps calcul optimisation : {duree_opt:.3f}s)",
                 fontsize=11, fontweight="bold")

    donnees = [
        ("X — Est/Ouest", dx, x_fgo, x_odom, rmse_gps_x, rmse_fgo_x, "tab:blue"),
        ("Y — Nord/Sud",  dy, y_fgo, y_odom, rmse_gps_y, rmse_fgo_y, "tab:orange"),
        ("Z — Altitude",  dz, z_fgo, z_odom, rmse_gps_z, rmse_fgo_z, "tab:green"),
    ]
    for ax, (label, brut, fgo, verite, r_gps, r_fgo, color) in zip(axes, donnees):
        ax.plot(t_gps, verite, "--", color="black", linewidth=1.3, label=f"{label} vérité (Odom)")
        ax.plot(t_gps, brut, "-", color=color, alpha=0.5, linewidth=1.2, label=f"{label} GPS brut")
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
    out_img = os.path.join(IMG_DIR, f"fgo_imu_gps_coupled_{nom}.pdf")
    plt.savefig(out_img)
    print(f"[FGO GPS+IMU] Figure sauvegardée : {out_img}")

    # ── AJOUT : trace des 6 composantes du biais IMU au fil du vol ─────────
    # (demande du prof : verifier que les biais convergent, pas juste
    # regarder la valeur finale)
    bias_acc = np.array([biais_finaux[k].accelerometer() for k in range(n)])
    bias_gyro = np.array([biais_finaux[k].gyroscope() for k in range(n)])

    fig_bias, axes_bias = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    fig_bias.suptitle(f"Convergence du biais IMU estime — {nom}", fontsize=11, fontweight="bold")
    labels_acc = ["bias_ax", "bias_ay", "bias_az"]
    labels_gyro = ["bias_gx", "bias_gy", "bias_gz"]
    for i in range(3):
        axes_bias[i, 0].plot(t_gps, bias_acc[:, i], color="tab:red")
        axes_bias[i, 0].set_ylabel(f"{labels_acc[i]} (m/s²)")
        axes_bias[i, 0].grid(True)
        axes_bias[i, 1].plot(t_gps, bias_gyro[:, i], color="tab:purple")
        axes_bias[i, 1].set_ylabel(f"{labels_gyro[i]} (rad/s)")
        axes_bias[i, 1].grid(True)
    axes_bias[0, 0].set_title("Accelerometer bias")
    axes_bias[0, 1].set_title("Gyroscope bias")
    axes_bias[-1, 0].set_xlabel("Temps (s)")
    axes_bias[-1, 1].set_xlabel("Temps (s)")
    plt.tight_layout()
    out_img_bias = os.path.join(IMG_DIR, f"fgo_imu_gps_coupled_{nom}_bias.pdf")
    plt.savefig(out_img_bias)
    print(f"[FGO GPS+IMU] Figure biais sauvegardée : {out_img_bias}")

    # ── AJOUT : trajectoire vue du dessus (X vs Y) ──────────────────────────
    # (cette session : ajout du temps de calcul dans le titre, comme demande)
    fig_top, ax_top = plt.subplots(figsize=(8, 8))
    ax_top.plot(x_odom, y_odom, "--", color="black", linewidth=1.3, label="Vérité (Odom)")
    ax_top.plot(dx, dy, "-", color="tab:blue", alpha=0.4, linewidth=1.0, label="GPS brut")
    ax_top.plot(x_fgo, y_fgo, "-", color="tab:red", linewidth=1.8, label="FGO GPS+IMU")
    ax_top.set_xlabel("X — Est/Ouest (m)")
    ax_top.set_ylabel("Y — Nord/Sud (m)")
    ax_top.set_title(f"Trajectoire vue du dessus — {nom}  (temps calcul optimisation : {duree_opt:.3f}s)",
                      fontweight="bold")
    ax_top.legend(fontsize=9)
    ax_top.grid(True)
    ax_top.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    out_img_top = os.path.join(IMG_DIR, f"fgo_imu_gps_coupled_{nom}_topdown.pdf")
    plt.savefig(out_img_top)
    print(f"[FGO GPS+IMU] Figure vue du dessus sauvegardée : {out_img_top}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f"fgo_imu_gps_coupled_{nom}.csv")
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
    print(f"[FGO GPS+IMU] CSV sauvegardé : {out_csv} ({n} lignes)")


def main():
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
        run_fgo_gps_imu(nom, f["gps"], f["imu"], f["odom"])

    plt.show()


if __name__ == "__main__":
    main()