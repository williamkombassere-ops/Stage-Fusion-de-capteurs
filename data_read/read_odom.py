#!/usr/bin/env python3
"""
read_odom.py
------------
Node ROS2 abonné au topic /crazyflie/odom (nav_msgs/Odometry).
Fournit la position réelle du drone dans Gazebo (vérité terrain).
Stocke : temps, position (x, y, z), orientation quaternion (qx, qy, qz, qw)
ET vitesse linéaire réelle (vx, vy, vz).

AJOUT (suite à une question du prof -- tester si l'erreur de vitesse
initiale v0 est la cause de la dérive pendant les coupures GPS) :
  Gazebo publie déjà la vitesse réelle dans le même message Odometry
  (msg.twist.twist.linear), exprimée dans le repère du corps du drone
  (convention standard nav_msgs/Odometry). On la sauvegarde ici en plus
  de la position, pour pouvoir l'utiliser comme v0 "verite terrain"
  dans le FGO, a la place de la difference finie sur 2 points GPS
  (fragile, cf. discussion precedente).

Le quaternion est nécessaire pour calculer les vraies vitesses angulaires
via la formule : omega = 2 * q* ⊗ dq/dt
Appelé depuis main.py via OdomReader().
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class OdomReader(Node):
    def __init__(self):
        super().__init__("odom_reader")

        # Liste qui accumule les positions, orientations et vitesses réelles
        self.data = []

        # Abonnement au topic odométrie
        self.sub = self.create_subscription(
            Odometry,
            "/crazyflie/odom",
            self._callback,
            10  # taille de la file d'attente
        )

    def _callback(self, msg):
        """Appelé à chaque message Odometry reçu."""
        pos  = msg.pose.pose.position      # position 3D
        quat = msg.pose.pose.orientation   # orientation quaternion
        lin  = msg.twist.twist.linear      # AJOUT : vitesse linéaire réelle
                                            # (repère du corps, convention
                                            # standard nav_msgs/Odometry)

        self.data.append({
            "t":  msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            "x":  pos.x,    # position X en mètres
            "y":  pos.y,    # position Y en mètres
            "z":  pos.z,    # position Z en mètres (altitude réelle)
            "qx": quat.x,   # quaternion — composante x
            "qy": quat.y,   # quaternion — composante y
            "qz": quat.z,   # quaternion — composante z
            "qw": quat.w,   # quaternion — composante scalaire w
            "vx": lin.x,    # AJOUT : vitesse linéaire X (m/s), repère du corps
            "vy": lin.y,    # AJOUT : vitesse linéaire Y (m/s), repère du corps
            "vz": lin.z,    # AJOUT : vitesse linéaire Z (m/s), repère du corps
        })