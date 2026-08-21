#!/usr/bin/env python3
"""
read_flow.py
------------
Nodes ROS2 pour le "Flow Deck" du Crazyflie, composé de 2 capteurs
distincts dans le SDF :
  - optical_flow_camera (type 'camera')  → /crazyflie/optical_flow_camera
  - tof_altitude        (type 'gpu_ray') → /crazyflie/tof_altitude

ATTENTION — point important :
  Le capteur optical_flow_camera est un capteur CAMÉRA générique dans Gazebo : il publie des IMAGES BRUTES (sensor_msgs/Image), pas un flux optique déjà calculé (dx, dy en pixels/s) comme le ferait un
  vrai PMW3901. Il n'y a pas de plugin dédié dans le SDF pour ça. On calcule donc nous-mêmes le déplacement entre deux images consécutives par corrélation de phase (cv2.phaseCorrelate), ce qui
  donne un décalage sous-pixel (dx, dy) par paire d'images — c'est une approximation raisonnable du flux optique, pas une réplique exacte du capteur physique.

  Le capteur tof_altitude est confirmé bridgé en sensor_msgs/LaserScan (un seul faisceau, angle_min=angle_max=0) — la mesure utile est
  ranges[0]. Elle peut valoir +inf si rien n'est détecté dans la plage du capteur (range_max=2.0m).

Deux classes exportées :
  - FlowReader : lit les images, calcule et stocke le flux optique (dx, dy)
  - TofReader  : lit l'altitude mesurée par le capteur ToF
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


class FlowReader(Node):
    """
    Abonné à /crazyflie/optical_flow_camera (sensor_msgs/Image). Calcule le déplacement (dx, dy) en pixels entre deux images consécutives par corrélation de phase, et stocke le résultat.
    """

    def __init__(self):
        super().__init__("flow_reader")
        self.data = []          # liste de {"t", "dx", "dy"} (pixels entre 2 images)
        self._image_precedente = None
        self._t_precedent = None

        self.sub = self.create_subscription(
            Image,
            "/crazyflie/optical_flow_camera",
            self._callback,
            10
        )

    def _image_vers_gris(self, msg: Image) -> np.ndarray:
        """
        Convertit un message Image (R8G8B8, 64x64 d'après le SDF) en
        tableau numpy en niveaux de gris, sans dépendance à cv_bridge.
        """
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    def _callback(self, msg: Image):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        gris = self._image_vers_gris(msg)

        if self._image_precedente is not None:
            # Corrélation de phase : décalage sous-pixel (dx, dy) entre
            # les deux images consécutives.
            (dx, dy), _ = cv2.phaseCorrelate(self._image_precedente, gris)
            self.data.append({"t": t, "dx": dx, "dy": dy})

        self._image_precedente = gris
        self._t_precedent = t


class TofReader(Node):
    """
    Abonné à /crazyflie/tof_altitude (sensor_msgs/LaserScan — confirmé par 'ros2 topic info', un seul faisceau car angle_min=angle_max=0). Stocke la distance mesurée au sol (altitude relative), issue de
    ranges[0]. Cette valeur peut être infinie (float('inf')) si rien n'est détecté dans la plage du capteur (range_max=2.0m d'après le SDF) — on la garde telle quelle plutôt que de la filtrer, pour ne
    pas masquer un vrai problème de portée pendant un vol à haute altitude.
    """

    def __init__(self):
        super().__init__("tof_reader")
        self.data = []          # liste de {"t", "range"}

        self.sub = self.create_subscription(
            LaserScan,
            "/crazyflie/tof_altitude",
            self._callback,
            10
        )

    def _callback(self, msg: LaserScan):
        self.data.append({
            "t":     msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            "range": msg.ranges[0] if msg.ranges else float("nan"),
        })