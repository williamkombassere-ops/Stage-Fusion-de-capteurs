#!/usr/bin/env python3
"""
collect.py
----------
Module utilitaire commun à tous les scripts de vol et de collecte.
Fournit les fonctions de démarrage ROS2, collecte des capteurs et sauvegarde CSV.

Fonctions exportées :
  - init_nodes()        : initialise ROS2 et les 3 nodes capteurs
  - spin_once(executor) : traite les callbacks ROS2 disponibles
  - stop_nodes(...)     : arrête proprement ROS2
  - save_all(nom, ...)  : sauvegarde les CSV avec le bon suffixe

Convention de nommage des CSV :
  nom="sol"      → gps.csv / imu.csv / odom.csv
  nom="vertical" → gps_vertical.csv / imu_vertical.csv / odom_vertical.csv
  nom="carre"    → gps_carre.csv / etc.

Flow Deck (optionnel) : si des instances FlowReader/TofReader sont
passées à save_all() via flow_node/tof_node, deux CSV supplémentaires
sont sauvegardés : flow_<nom>.csv (t, dx, dy) et tof_<nom>.csv
(t, range).

MISE A JOUR : odom.csv contient maintenant aussi vx,vy,vz (vitesse
lineaire reelle publiee par Gazebo dans msg.twist.twist.linear, cf.
read_odom.py) -- ajoutee suite a une question du prof pour tester si
l'erreur de vitesse initiale v0 (difference finie sur 2 points GPS)
est la cause de la derive pendant les coupures GPS.
"""

import os
import csv

import rclpy
from rclpy.executors import MultiThreadedExecutor

from data_read.read_gps  import GpsReader
from data_read.read_imu  import ImuReader
from data_read.read_odom import OdomReader

import os as _os
CSV_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "csv_files")


def save_csv(filepath: str, fieldnames: list, rows: list):
    """Sauvegarde une liste de dicts dans un fichier CSV."""
    os.makedirs(CSV_DIR, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] Sauvegardé : {filepath} ({len(rows)} lignes)")


def init_nodes(extra_nodes=None):
    """
    Initialise ROS2 et crée les 3 nodes de lecture capteurs.
    Retourne (executor, gps_node, imu_node, odom_node).

    extra_nodes : liste de nodes supplémentaires à ajouter à l'executor
                  (ex: node de contrôle Twist pour les scripts de vol,
                  ou FlowReader/TofReader pour le Flow Deck)
    """
    rclpy.init()

    gps_node  = GpsReader()
    imu_node  = ImuReader()
    odom_node = OdomReader()

    executor = MultiThreadedExecutor()
    executor.add_node(gps_node)
    executor.add_node(imu_node)
    executor.add_node(odom_node)

    # Ajout des nodes supplémentaires (contrôle, Flow Deck, etc.)
    if extra_nodes:
        for node in extra_nodes:
            executor.add_node(node)

    return executor, gps_node, imu_node, odom_node


def spin_once(executor, timeout_sec=0.05):
    """Traite les callbacks ROS2 disponibles (non bloquant)."""
    executor.spin_once(timeout_sec=timeout_sec)


def stop_nodes(executor, *nodes):
    """Arrête proprement l'executor et détruit tous les nodes."""
    executor.shutdown()
    for node in nodes:
        node.destroy_node()
    rclpy.shutdown()


def save_all(nom: str, gps_node, imu_node, odom_node,
             flow_node=None, tof_node=None):
    """
    Sauvegarde les CSV pour la trajectoire donnée.
    Convention : nom="sol" → gps.csv / nom="vertical" → gps_vertical.csv

    flow_node, tof_node : optionnels — si fournis (instances FlowReader
    et TofReader depuis data_read/read_flow.py), sauvegarde aussi
    flow_<nom>.csv et tof_<nom>.csv.
    """
    suf = "" if nom == "sol" else f"_{nom}"

    msg_compte = (f"[Collect] GPS:{len(gps_node.data)} | "
                  f"IMU:{len(imu_node.data)} | "
                  f"Odom:{len(odom_node.data)}")
    if flow_node is not None:
        msg_compte += f" | Flow:{len(flow_node.data)}"
    if tof_node is not None:
        msg_compte += f" | ToF:{len(tof_node.data)}"
    print(msg_compte + " messages")

    save_csv(os.path.join(CSV_DIR, f"gps{suf}.csv"),
             ["t", "lat", "lon", "alt"],
             gps_node.data)

    save_csv(os.path.join(CSV_DIR, f"imu{suf}.csv"),
             ["t", "ax", "ay", "az", "wx", "wy", "wz"],
             imu_node.data)

    # MISE A JOUR : vx, vy, vz ajoutees (vitesse reelle Gazebo, cf.
    # read_odom.py) -- necessaire pour tester v0 "verite terrain"
    save_csv(os.path.join(CSV_DIR, f"odom{suf}.csv"),
             ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz"],
             odom_node.data)

    # ── Flow Deck (optionnel) ───────────────────────────────────────────
    if flow_node is not None:
        save_csv(os.path.join(CSV_DIR, f"flow{suf}.csv"),
                 ["t", "dx", "dy"],
                 flow_node.data)

    if tof_node is not None:
        save_csv(os.path.join(CSV_DIR, f"tof{suf}.csv"),
                 ["t", "range"],
                 tof_node.data)