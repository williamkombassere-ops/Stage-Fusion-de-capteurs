#!/usr/bin/env python3
"""
read_imu.py
-----------
Node ROS2 abonné au topic /crazyflie/imu (sensor_msgs/Imu).
Stocke à chaque message : temps, accélérations linéaires (ax,ay,az)
et vitesses angulaires (wx,wy,wz).
Appelé depuis main.py via ImuReader().
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuReader(Node):
    def __init__(self):
        super().__init__("imu_reader")

        # Liste qui accumule les mesures IMU au fil du temps
        self.data = []

        # Abonnement au topic IMU
        self.sub = self.create_subscription(
            Imu,
            "/crazyflie/imu",
            self._callback,
            10  # taille de la file d'attente
        )

    def _callback(self, msg):
        """Appelé à chaque message IMU reçu."""
        self.data.append({
            "t":  msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,  # temps absolu (s)
            # Accélérations linéaires (m/s²) — incluent la gravité au repos
            "ax": msg.linear_acceleration.x,
            "ay": msg.linear_acceleration.y,
            "az": msg.linear_acceleration.z,
            # Vitesses angulaires (rad/s)
            "wx": msg.angular_velocity.x,
            "wy": msg.angular_velocity.y,
            "wz": msg.angular_velocity.z,
        })