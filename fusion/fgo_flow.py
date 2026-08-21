#!/usr/bin/env python3
"""
fgo_flow.py
-----------
Estime la position du drone à partir du FLOW DECK SEUL (flux optique +
capteur ToF d'altitude), sans GPS ni IMU.

Deux logiques très différentes selon l'axe :
  - Z (altitude) : mesure ABSOLUE directe du ToF (ranges[0] du
    LaserScan) — bornée, pas d'intégration, pas de dérive (comme le
    GPS pour l'altitude).
  - X, Y (horizontal) : le flux optique ne donne qu'un DÉPLACEMENT
    RELATIF en pixels entre deux images consécutives — il faut le
    convertir en mètres via la hauteur au sol (ToF) et le champ de
    vision de la caméra, puis CUMULER ces déplacements dans le temps.
    Comme pour l'IMU, aucune référence absolue ne recale cette
    intégration : la position horizontale dérive avec le temps.

Conversion pixel → mètre (géométrie de projection en perspective) :
    déplacement_m = déplacement_px * (2 * h * tan(HFOV/2) / largeur_px)
où h est la hauteur au sol (mètres), HFOV le champ de vision
horizontal de la caméra (rad, 1.047 dans le SDF), largeur_px la
largeur de l'image en pixels (64 dans le SDF).

Usage :
  python3 fusion/fgo_flow.py vertical
  python3 fusion/fgo_flow.py               ← détecte toutes les trajectoires

CSV attendus (produits par data_read/read_flow.py, à sauvegarder via
collect.py — voir save_all) :
  csv_files/flow_<traj>.csv  : colonnes t, dx, dy   (pixels entre 2 images)
  csv_files/tof_<traj>.csv   : colonnes t, range    (mètres, peut être inf)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_scripts.plot_sensors import _plot_axe

_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_DIR = os.path.join(_ROOT, "csv_files")
FLOW_DIR = os.path.join(CSV_DIR, "flow_seul_csv")
IMG_DIR  = os.path.join(_ROOT, "images")

# ─── Paramètres caméra (depuis le SDF) ────────────────────────────────────────
HFOV        = 1.047   # champ de vision horizontal (rad) — 60°, cf. SDF
LARGEUR_PX  = 64       # largeur de l'image (pixels) — cf. SDF
Z_MIN_ECHELLE = 0.05   # borne basse (m) pour éviter l'explosion de l'échelle
                       # pixel->metre quand z est proche de 0 (decollage/
                       # atterrissage), voir clip plus bas
# ─────────────────────────────────────────────────────────────────────────────


def estimer_position_flow(flow_csv: str, tof_csv: str) -> dict:
    """
    Calcule la trajectoire (t, x, y, z) à partir du flux optique et du
    ToF seuls.
    """
    df_flow = pd.read_csv(flow_csv).sort_values("t").reset_index(drop=True)
    df_tof  = pd.read_csv(tof_csv).sort_values("t").reset_index(drop=True)

    t0 = df_flow["t"].iloc[0]
    t_flow = (df_flow["t"] - t0).to_numpy()
    t_tof  = (df_tof["t"] - t0).to_numpy()

    # ── Z : mesure absolue directe du ToF, interpolée sur les temps du flow ──
    # Nettoyage préalable : le ToF renvoie ±inf quand l'altitude est hors de
    # sa plage valide (range.min=0.02m au décollage/atterrissage, cf. SDF).
    # On remplace ces trous par interpolation linéaire sur les valeurs
    # finies voisines avant de réinterpoler sur les temps du flow.
    range_tof_brut = df_tof["range"].to_numpy()
    valide = np.isfinite(range_tof_brut)
    if not valide.all():
        n_invalides = int((~valide).sum())
        print(f"[Flow seul] {n_invalides} valeurs ToF non finies nettoyées par interpolation")
        range_tof = np.interp(t_tof, t_tof[valide], range_tof_brut[valide])
    else:
        range_tof = range_tof_brut
    z = np.interp(t_flow, t_tof, range_tof)

    # Sécurité : évite une échelle pixel->métre qui explose si z est
    # anormalement proche de 0 (décollage/atterrissage)
    z_pour_echelle = np.clip(z, Z_MIN_ECHELLE, None)

    # ── X, Y : conversion pixel → mètre à chaque pas, puis cumul ───────────
    dx_px = df_flow["dx"].to_numpy()
    dy_px = df_flow["dy"].to_numpy()

    # échelle mètre/pixel, dépend de la hauteur courante (via ToF interpolé)
    # tan(HFOV/2) : demi-angle de champ de vision
    echelle = (2.0 * z_pour_echelle * np.tan(HFOV / 2.0)) / LARGEUR_PX
    dx_m = dx_px * echelle
    dy_m = dy_px * echelle

    x = np.concatenate(([0], np.cumsum(dx_m[1:])))
    y = np.concatenate(([0], np.cumsum(dy_m[1:])))

    return {"t": t_flow, "x": x, "y": y, "z": z}


def run_flow_seul(nom: str, flow_csv: str, tof_csv: str, odom_csv: str):
    """
    Calcule la position par Flow Deck seul, trace le résultat vs Odom
    (position + erreur dans le même graphique par axe), et sauvegarde
    le résultat dans un CSV.
    """
    print(f"\n[Flow seul] Trajectoire : {nom}")
    est = estimer_position_flow(flow_csv, tof_csv)

    df_odom = pd.read_csv(odom_csv).sort_values("t")
    t_odom  = (df_odom["t"] - df_odom["t"].iloc[0]).to_numpy()
    x_odom  = np.interp(est["t"], t_odom, df_odom["x"].to_numpy())
    y_odom  = np.interp(est["t"], t_odom, df_odom["y"].to_numpy())
    z_odom  = np.interp(est["t"], t_odom, df_odom["z"].to_numpy())

    rmse = lambda a, b: np.sqrt(np.nanmean((a - b) ** 2))
    print(f"[Flow seul] RMSE X = {rmse(est['x'], x_odom):.4f} m")
    print(f"[Flow seul] RMSE Y = {rmse(est['y'], y_odom):.4f} m")
    print(f"[Flow seul] RMSE Z = {rmse(est['z'], z_odom):.4f} m")

    # ── Tracé : 3 sous-graphes (X, Y, Z), position Flow Deck + vérité ──────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Position Flow Deck seul vs Odom — {nom}",
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
    out_img = os.path.join(IMG_DIR, f"flow_seul_{nom}.pdf")
    plt.savefig(out_img)
    print(f"[Flow seul] Figure sauvegardée : {out_img}")

    # ── Sauvegarde CSV ──────────────────────────────────────────────────
    os.makedirs(FLOW_DIR, exist_ok=True)
    out_csv = os.path.join(FLOW_DIR, f"flow_seul_{nom}.csv")
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
    print(f"[Flow seul] CSV sauvegardé : {out_csv}")


def detecter_trajectoires():
    """Détecte les groupes flow/tof/Odom disponibles dans csv_files/."""
    import re
    trajectoires = {}
    for f in os.listdir(CSV_DIR):
        m = re.match(r"^flow(?:_(.+))?\.csv$", f)
        if m:
            nom  = m.group(1) or "sol"
            suf  = f"_{nom}" if nom != "sol" else ""
            flow = os.path.join(CSV_DIR, f"flow{suf}.csv")
            tof  = os.path.join(CSV_DIR, f"tof{suf}.csv")
            odom = os.path.join(CSV_DIR, f"odom{suf}.csv")
            if all(os.path.isfile(p) for p in [flow, tof, odom]):
                trajectoires[nom] = {"flow": flow, "tof": tof, "odom": odom}
    return trajectoires


def main():
    trajectoires = detecter_trajectoires()
    if not trajectoires:
        print("[Erreur] Aucun groupe flow/tof/Odom trouvé dans csv_files/")
        print("[Info]   Vérifie que collect.py sauvegarde bien flow_<traj>.csv "
              "et tof_<traj>.csv (via FlowReader/TofReader).")
        return

    if len(sys.argv) > 1:
        nom = sys.argv[1]
        if nom not in trajectoires:
            print(f"[Erreur] Trajectoire '{nom}' introuvable. "
                  f"Disponibles : {list(trajectoires.keys())}")
            return
        trajectoires = {nom: trajectoires[nom]}

    for nom, f in trajectoires.items():
        run_flow_seul(nom, f["flow"], f["tof"], f["odom"])

    plt.show()


if __name__ == "__main__":
    main()