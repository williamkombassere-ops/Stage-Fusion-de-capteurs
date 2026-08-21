#!/usr/bin/env python3
"""
plot_sensors.py
---------------
Détecte automatiquement tous les groupes de CSV dans csv_files/et génère 3 images par trajectoire :
  - gps_<traj>.pdf (ou fgo_gps_<traj>.pdf / fgo_all_<traj>.pdf selon le contenu disponible) : GPS vs Odom (X, Y, Z) + erreur, avec les courbes FGO GPS et/ou FGO GPS+IMU superposées si elles existent.
  - imu_accel_<traj>.pdf : IMU accélérations (ax, ay, az) + vérité terrain + erreur
  - imu_gyro_<traj>.pdf  : IMU vitesses angulaires (wx, wy, wz) + vérité terrain + erreur
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── Constantes globales ──────────────────────────────────────────────────────
# _ROOT : racine du projet, calculée depuis l'emplacement de ce fichier (donc valide peu importe d'où le script est lancé).
# R_TERRE : rayon terrestre moyen, nécessaire pour convertir des degrés GPS (lat/lon) en un déplacement en mètres.
# GRAVITY : utilisée comme vérité terrain de l'accéléromètre au repos(un capteur immobile "sent" la réaction du sol contre de gravité, donc az_théorique = +9.8 m/s² et non 0).
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.join(_HERE, "..")
R_TERRE = 6_371_000.0
GRAVITY = 9.8

# DÉTECTION DES TRAJECTOIRES DISPONIBLES
def detecter_groupes(csv_dir: str) -> dict:
    """Convention de nommage des CSV :
      gps.csv / imu.csv / odom.csv          → préfixe = "sol"
      gps_vertical.csv / imu_vertical.csv   → préfixe = "vertical"
      gps_rotation.csv / imu_rotation.csv   → préfixe = "rotation"
    """
    if not os.path.isdir(csv_dir):
        print(f"[Erreur] Dossier introuvable : {csv_dir}")
        return {}

    # Étape 1 : repérer tous les préfixes de trajectoire à partir des
    fichiers = os.listdir(csv_dir)
    prefixes = set()
    for f in fichiers:
        m = re.match(r"^gps(?:_(.+))?\.csv$", f)
        if m:
            prefixes.add(m.group(1) or "sol")

    # Étape 2 : pour chaque préfixe, vérifier que les 3 fichiers capteurs
    groupes = {}
    for p in sorted(prefixes):
        suffix = f"_{p}" if p != "sol" else ""
        gps  = os.path.join(csv_dir, f"gps{suffix}.csv")
        imu  = os.path.join(csv_dir, f"imu{suffix}.csv")
        odom = os.path.join(csv_dir, f"odom{suffix}.csv")

        if all(os.path.isfile(f) for f in [gps, imu, odom]):
            groupes[p] = {
                "gps": gps, "imu": imu, "odom": odom,
                "statique": (p == "sol"),
            }
            print(f"[Info] Groupe détecté : '{p}' → {os.path.basename(gps)}")
        else:
            print(f"[Avertissement] Groupe '{p}' incomplet — ignoré")

    return groupes

# CONVERSIONS ET VÉRITÉ TERRAIN
def gps_to_meters(lat, lon, alt):
    """Convertit des tableaux lat/lon/alt (degrés) en déplacement (X, Y, Z) en mètres, par rapport à une origine locale.
    """
    lat_rad = np.radians(lat)
    N_REF   = min(10, len(lat))
    lat0    = np.median(lat_rad[:N_REF])
    lon0    = np.median(lon[:N_REF])
    alt0    = np.median(alt[:N_REF])

    dx = (lon - lon0) * np.cos(lat0) * np.radians(R_TERRE)
    dy = (lat - np.median(lat[:N_REF])) * np.radians(R_TERRE)
    dz = alt - alt0
    return dx, dy, dz

def odom_to_imu_truth(odom_csv: str, t_imu: np.ndarray) -> dict:
    """
    -Calcule ce que l'IMU DEVRAIT mesurer à partir de la position/orientation réelle simulée (Odom, sans bruit).
    -Accélération vraie : dérivée seconde de la position + gravité sur Z 
    -Vitesse angulaire vraie : formule quaternionique exacte   omega = 2 * q* ⊗ dq/dt
    (q* = conjugué du quaternion, dq/dt = sa dérivée numérique).Physiquement correcte contrairement à une dérivée de position, qui ne capte pas une rotation pure (le drone peut tourner sur lui-même sans que sa position ne change).
    """
    df = pd.read_csv(odom_csv).sort_values("t")
    t  = df["t"].to_numpy()

    # ── Accélération : double dérivée numérique de la position ───────────
    ax = np.gradient(np.gradient(df["x"].to_numpy(), t), t)
    ay = np.gradient(np.gradient(df["y"].to_numpy(), t), t)
    az = np.gradient(np.gradient(df["z"].to_numpy(), t), t) + GRAVITY

    # ── Vitesse angulaire : dérivée du quaternion, convertie via produit
    if "qx" in df.columns:
        qx, qy, qz, qw = (df[c].to_numpy() for c in ("qx", "qy", "qz", "qw"))
        dqx, dqy, dqz, dqw = (np.gradient(c, t) for c in (qx, qy, qz, qw))

        # omega_pur = 2 * q* ⊗ dq/dt , avec q* = (-qx, -qy, -qz, qw)
        wx = 2 * ( qw*dqx - qz*dqy + qy*dqz - qx*dqw) * (-1)
        wy = 2 * ( qz*dqx + qw*dqy - qx*dqz - qy*dqw) * (-1)
        wz = 2 * (-qy*dqx + qx*dqy + qw*dqz - qz*dqw) * (-1)
    else:
        print("[Avertissement] Colonnes quaternion absentes — relancer la collecte")
        wx = wy = wz = np.zeros_like(t)

    # Interpolation sur les timestamps exacts de l'IMU pour pouvoir comparer point par point avec les mesures brutes
    t_rel = t - t[0]
    return {col: np.interp(t_imu, t_rel, data)
            for col, data in zip(["ax", "ay", "az", "wx", "wy", "wz"],
                                  [ax, ay, az, wx, wy, wz])}

# TRACÉ COMMUN : mesuré + vérité + erreur, tous dans LE MÊME sous-graphe
# On trace l'erreur (mesuré - vérité) comme une 3ème courbe superposée
# aux courbes mesuré/vérité, pas dans un sous-graphe séparé — ça évite
# la confusion visuelle de duplication quand la vérité est proche de 0
# (dans ce cas l'erreur ressemble à la mesure, ce qui est normal et pas
# un bug : erreur = mesuré - 0 = mesuré).
def _rmse(a, b):
    """Erreur quadratique moyenne entre deux tableaux de même longueur."""
    return np.sqrt(np.mean((a - b) ** 2))


def _plot_axe(ax, t, mesure, verite, label, unite, color, courbes_extra=None):
    """
    Trace mesuré, vérité et erreur (mesuré - vérité) sur UN SEUL sous-graphe,
    plus un encart texte avec la valeur RMSE exacte (notation scientifique
    .2e, pas arrondie à 100 près — un .4f classique arrondirait par exemple
    tout ce qui est < 5e-5 à "0.0000").
    courbes_extra : liste de dicts {"label", "color", "t", "data"} (utilisé par fig1 pour les courbes FGO — leur erreur et leur RMSE sont tracés aussi)
    """
    courbes_extra = courbes_extra or []

    ax.plot(t, mesure, color=color, linewidth=1.0, alpha=0.9, label=f"{label} mesuré")
    ax.plot(t, verite, color="black", linewidth=1.6, linestyle="--",
            label=f"{label} vérité")

    erreur   = mesure - verite
    rmse_val = _rmse(mesure, verite)
    ax.plot(t, erreur, color=color, linewidth=1.0, linestyle=":",
            label=f"{label} erreur (mesuré-vérité)")

    rmse_txt = f"RMSE {label} = {rmse_val:.2e} {unite}"

    for c in courbes_extra:
        ax.plot(c["t"], c["data"], color=c["color"], linewidth=1.3,
                label=c["label"])
        data_interp   = np.interp(t, c["t"], c["data"])
        erreur_extra  = data_interp - verite
        rmse_extra    = _rmse(data_interp, verite)
        ax.plot(t, erreur_extra, color=c["color"], linewidth=1.0, linestyle=":",
                label=f"{c['label']} erreur")
        rmse_txt += f"\nRMSE {c['label']} = {rmse_extra:.2e} {unite}"

    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_ylabel(f"{label} ({unite})")
    ax.grid(True)
    ax.text(0.01, 0.97, rmse_txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.85))
    ax.legend(loc="upper right", fontsize=7)

# FIGURE 1 : GPS vs ODOM (+ FGO optionnel)
def _charger_courbe_fgo(fgo_csv, label, color):
    """
    Charge un CSV de résultat FGO (colonnes t,x,y,z) s'il existe, et renvoie un petit dict prêt à l'emploi pour le tracé. Renvoie None si le fichier n'existe pas — pour ne PAS dupliquer un bloc "if has_fgo: ... if has_fgo_imu: ..." pour chaque cas.
    """
    if fgo_csv is None or not os.path.isfile(fgo_csv):
        return None
    df = pd.read_csv(fgo_csv).sort_values("t")
    return {
        "label": label, "color": color,
        "t": df["t"].to_numpy(),
        "x": df["x"].to_numpy(), "y": df["y"].to_numpy(), "z": df["z"].to_numpy(),
    }

def fig1_gps_vs_odom(gps_csv: str, odom_csv: str, titre: str,
                      fgo_csv: str = None, fgo_imu_csv: str = None):
    """
    Figure 1 : GPS vs Odom (X, Y, Z) + erreur, dans le même graphique par axe. Ajoute automatiquement les courbes FGO GPS et/ou FGO GPS+IMU si les CSV correspondants existent (fgo_csv, fgo_imu_csv).
    """
    # ── Chargement GPS + conversion en mètres ─────────────────────────────
    df_gps  = pd.read_csv(gps_csv).sort_values("t")
    df_odom = pd.read_csv(odom_csv).sort_values("t")
    dx, dy, dz = gps_to_meters(df_gps["lat"].to_numpy(),
                                df_gps["lon"].to_numpy(),
                                df_gps["alt"].to_numpy())

    # ── Alignement temporel GPS/Odom + recalage sur le repère Odom ────────
    t0    = df_gps["t"].iloc[0]
    t_gps = (df_gps["t"] - t0).to_numpy()
    t_od  = (df_odom["t"] - t0).to_numpy()
    x_od  = np.interp(t_gps, t_od, df_odom["x"].to_numpy())
    y_od  = np.interp(t_gps, t_od, df_odom["y"].to_numpy())
    z_od  = np.interp(t_gps, t_od, df_odom["z"].to_numpy())
    dx += x_od[0]; dy += y_od[0]; dz += z_od[0]

    # ── Correction du biais GPS (X, Y) sur les 5 premières mesures ────────
    # Le GPS peut avoir un léger décalage systématique par rapport à la
    # vérité terrain ; on l'estime au début (drone encore quasi immobile)
    # et on le soustrait à toute la trajectoire.
    N_BIAS = 5
    dx -= np.mean(dx[:N_BIAS] - x_od[:N_BIAS])
    dy -= np.mean(dy[:N_BIAS] - y_od[:N_BIAS])

    # ── Chargement des courbes FGO (GPS seul et/ou GPS+IMU) si présentes ──
    # Chacune est un dict ou None — le tracé plus bas boucle simplement
    # sur la liste des courbes non-None, sans dupliquer de code par cas.
    fgo_gps_c = _charger_courbe_fgo(fgo_csv,     "FGO GPS",     "tab:red")
    fgo_imu_c = _charger_courbe_fgo(fgo_imu_csv, "FGO GPS+IMU", "tab:purple")
    courbes_fgo = [c for c in [fgo_gps_c, fgo_imu_c] if c is not None]

    # ── Tracé : 3 sous-graphes (X, Y, Z), mesuré+vérité+erreur ensemble ───
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"GPS vs Odom — {titre}", fontsize=11, fontweight="bold")

    donnees = [
        ("X — Est/Ouest", "tab:blue",   dx, x_od, "x"),
        ("Y — Nord/Sud",  "tab:orange", dy, y_od, "y"),
        ("Z — Altitude",  "tab:green",  dz, z_od, "z"),
    ]

    for ax, (label, color, mesure, verite, cle) in zip(axes, donnees):
        extra = [{"label": c["label"], "color": c["color"], "t": c["t"], "data": c[cle]}
                  for c in courbes_fgo]
        _plot_axe(ax, t_gps, mesure, verite, label, "m", color, courbes_extra=extra)

    axes[-1].set_xlabel("Temps (s)")
    plt.tight_layout()

    # ── Sauvegarde : nom de fichier selon les courbes FGO présentes ───────
    if fgo_gps_c and fgo_imu_c:
        prefix = "fgo_all"
    elif fgo_imu_c:
        prefix = "fgo_gps_imu"
    elif fgo_gps_c:
        prefix = "fgo_gps"
    else:
        prefix = "gps"

    nom = os.path.basename(gps_csv).replace("gps_", "").replace("gps", "sol").replace(".csv", "")
    out = os.path.join(_ROOT, "images", f"{prefix}_{nom}.pdf")
    plt.savefig(out)
    print(f"[Plot] Figure 1 sauvegardée : {out}")

# FIGURES 2 & 3 : IMU (accélérations et vitesses angulaires)
def _fig_imu_generique(imu_csv: str, odom_csv: str, statique: bool, titre: str, colonnes: list, unite: str, titre_figure: str, prefixe_fichier: str, verite_statique: dict):
    df = pd.read_csv(imu_csv)
    t  = (df["t"] - df["t"].iloc[0]).to_numpy()

    # Vérité terrain : théorique si statique, sinon dérivée de l'Odom réel
    if statique:
        truth = {c: np.full_like(t, verite_statique[c]) for c in colonnes}
    else:
        raw   = odom_to_imu_truth(odom_csv, t)
        truth = {c: raw[c] for c in colonnes}

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{titre_figure} — {titre}", fontsize=13, fontweight="bold")

    couleurs = ["tab:blue", "tab:orange", "tab:green"]
    for ax, col, color in zip(axes, colonnes, couleurs):
        _plot_axe(ax, t, df[col].to_numpy(), truth[col], col, unite, color)

    axes[-1].set_xlabel("Temps (s)")
    plt.tight_layout()

    nom = os.path.basename(imu_csv).replace("imu_", "").replace("imu", "sol").replace(".csv", "")
    out = os.path.join(_ROOT, "images", f"{prefixe_fichier}_{nom}.pdf")
    plt.savefig(out)
    print(f"[Plot] {titre_figure} sauvegardée : {out}")


def fig2_imu_accel(imu_csv: str, odom_csv: str, statique: bool, titre: str):
    """Figure 2 : accélérations IMU (ax, ay, az) — sauvegardée en imu_accel_<titre>.pdf"""
    _fig_imu_generique(
        imu_csv, odom_csv, statique, titre,
        colonnes=["ax", "ay", "az"], unite="m/s²",
        titre_figure="IMU — Accélérations", prefixe_fichier="imu_accel",
        verite_statique={"ax": 0.0, "ay": 0.0, "az": GRAVITY},
    )


def fig3_imu_gyro(imu_csv: str, odom_csv: str, statique: bool, titre: str):
    """Figure 3 : vitesses angulaires IMU (wx, wy, wz) — sauvegardée en imu_gyro_<titre>.pdf"""
    _fig_imu_generique(
        imu_csv, odom_csv, statique, titre,
        colonnes=["wx", "wy", "wz"], unite="rad/s",
        titre_figure="IMU — Vitesses angulaires", prefixe_fichier="imu_gyro",
        verite_statique={"wx": 0.0, "wy": 0.0, "wz": 0.0},
    )

# FONCTION PRINCIPALE — appelée par les scripts de vol et en standalone
def plot_all(csv_dir: str):
    """
    Détecte toutes les trajectoires complètes (gps+imu+odom) dans csv_dir et génère les 3 images (gps, imu_accel, imu_gyro) pour chacune.
    """
    groupes = detecter_groupes(csv_dir)
    os.makedirs(os.path.join(_ROOT, "images"), exist_ok=True)

    if not groupes:
        print("[Avertissement] Aucun groupe CSV complet trouvé.")
        return

    for nom, g in groupes.items():
        print(f"\n[Plot] Trajectoire : {nom}")
        fig1_gps_vs_odom(g["gps"], g["odom"], titre=nom)
        fig2_imu_accel(g["imu"], g["odom"], g["statique"], titre=nom)
        fig3_imu_gyro(g["imu"], g["odom"], g["statique"], titre=nom)

    plt.show()

# ─── Lancement standalone : python3 plot_scripts/plot_sensors.py ────────────
if __name__ == "__main__":
    csv_dir = os.path.join(_ROOT, "csv_files")
    plot_all(csv_dir)