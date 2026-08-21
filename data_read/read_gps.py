#!/usr/bin/env python3
"""
read_gps.py
-----------
Node ROS2 abonné au topic /crazyflie/gps (sensor_msgs/NavSatFix). Stocke chaque message reçu dans self.data sous forme de dict. Appelé depuis les scripts de vol via GpsReader().

Simulation de perte GPS ("méthode Gazebo") :
  Ce node simule automatiquement une ou plusieurs coupures GPS, démarrant
  a des instants fixes après le premier message GPS reçu. C'est entièrement
  autonome — AUCUN script de vol n'a besoin de savoir ou de configurer quoi
  que ce soit : n'importe quel vol qui utilise GpsReader() aura automatiquement
  ce(s) trou(s) de données GPS, peu importe sa durée ou ses phases.

  Le capteur navsat de Gazebo continue de calculer et publier des mesures en
  interne — ce node choisit de les IGNORER pendant la ou les fenêtres de
  coupure. Les messages arrivant dans ces fenêtres ne sont jamais stockés,
  comme si le signal GPS était réellement absent — contrairement à un
  filtrage a posteriori du CSV, qui ne fait qu'effacer des lignes après coup.

  DEUX CONFIGURATIONS DISPONIBLES CI-DESSOUS (dans _callback) :
  seule UNE des deux doit être active a la fois (l'autre commentee) :
    - Bloc A : 1 coupure unique de BLACKOUT_DUREE secondes.
    - Bloc B : N coupures de BLACKOUT_DUREE_COURTE secondes chacune,
      demarrant aux instants de BLACKOUT_DEBUTS.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

# ─── Bloc A : coupure GPS unique ──────────────────────────────────────────
# Entièrement définis ici — aucun script de vol n'a besoin d'y toucher.
BLACKOUT_DEBUT = 5.0   # démarre 5s après le premier message GPS reçu
BLACKOUT_DUREE = 6.0   # durée de la coupure (s)

# ─── Bloc B : N coupures GPS courtes ──────────────────────────────────────
# Décommenter le bloc correspondant dans _callback pour activer celle-ci
# (et commenter le Bloc A). Instants de départ relatifs au 1er message GPS.
BLACKOUT_DEBUTS       = [2.0, 7.0, 12.0]   # 3 coupures, cycle 2s GPS / 3s coupure
BLACKOUT_DUREE_COURTE = 3.0                 # durée de chaque coupure (s)

# ─── Bloc C : GPS actif seulement au tout début, puis coupure totale ──────
# GPS disponible de 0 à GPS_FENETRE_FIN secondes, puis plus aucun message
# GPS pour le reste du vol (dead-reckoning IMU pur ensuite, pour voir la
# dérive du bruit IMU seul sans plus jamais de recalage GPS).
GPS_FENETRE_FIN = 3.0   # GPS actif de 0s a cette valeur (s), coupé au-dela


class GpsReader(Node):
    def __init__(self):
        super().__init__("gps_reader")

        # Liste qui accumule les mesures GPS au fil du temps
        self.data = []

        # ── Fenêtre(s) de coupure GPS (perte de signal simulée) ───────────
        # Fixée dès la construction du node : active par défaut pour tout
        # vol, sans configuration externe.
        self._t0 = None
        self._blackout = (BLACKOUT_DEBUT, BLACKOUT_DEBUT + BLACKOUT_DUREE)

        # Abonnement au topic GPS
        self.sub = self.create_subscription(
            NavSatFix,
            "/crazyflie/gps",
            self._callback,
            10  # taille de la file d'attente
        )

    def _callback(self, msg):
        """Appelé à chaque message GPS reçu. Ignore le message si on est dans une fenêtre de perte GPS simulée (blackout)."""
        t_abs = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self._t0 is None:
            self._t0 = t_abs
        t_rel = t_abs - self._t0

        # ── BLOC A : ICI JE PEUX ACTIVER ET DESACTIVER LA COUPURE UNIQUE (6s) ──
        # debut, fin = self._blackout
        # if debut <= t_rel <= fin:
        #     # Perte GPS simulée : message reçu mais volontairement jeté, comme si le capteur n'avait rien transmis.
        #     return

        #── BLOC B : ICI JE PEUX ACTIVER/DESACTIVER LES 4 COUPURES DE 1s ───────
        # for debut in BLACKOUT_DEBUTS:
        #     fin = debut + BLACKOUT_DUREE_COURTE
        #     if debut <= t_rel <= fin:
        #         # Perte GPS simulée : message reçu mais volontairement jeté, comme si le capteur n'avait rien transmis.
        #         return

        # ── BLOC C : ICI JE PEUX ACTIVER/DESACTIVER LA FENETRE UNIQUE (0-3s) ──
        # if t_rel > GPS_FENETRE_FIN:
        #     # Perte GPS simulée : message reçu mais volontairement jeté, comme si le capteur n'avait rien transmis.
        #     return

        self.data.append({
            "t":   t_abs,                  # temps absolu (s)
            "lat": msg.latitude,   # latitude en degrés
            "lon": msg.longitude,  # longitude en degrés
            "alt": msg.altitude,   # altitude en mètres
        })