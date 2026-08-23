Fusion de Capteurs GPS – IMU pour Drone Autonome

Projet réalisé dans le cadre d'un stage Erasmus au laboratoire RE.PLAN de l'Université Polytechnique de Bucarest (UPB), Roumanie.

Objectif

Développer une méthode de localisation robuste pour un drone autonome en fusionnant plusieurs capteurs (GPS, IMU, et à terme Optical Flow), afin de pallier les limites de chaque capteur pris isolément :

le GPS seul est fortement bruité et sujet aux trajets multiples (multipath) en milieu urbain, voire totalement absent en intérieur ou en tunnel ;
l'IMU seul accumule une dérive (drift) au fil du temps.
Approche : Factor Graph Optimization (FGO)

Plutôt qu'un filtre de Kalman étendu classique (qui linéarise localement et peut diverger), le projet s'appuie sur l'optimisation par graphe de facteurs (FGO) via la bibliothèque GTSAM. Cette approche traite l'estimation comme un problème de moindres carrés non-linéaire global, en ré-optimisant l'ensemble de la trajectoire à partir de :

un facteur a priori (état initial) ;
des facteurs GPS (contrainte de position absolue) ;
des facteurs IMU (préintégration des mesures inertielles entre deux instants) ;
des facteurs de biais (modélisation de la dérive de l'IMU).
Stack technique
ROS2 — communication entre les nœuds de simulation et de traitement
Gazebo — simulation d'un drone Crazyflie en environnement virtuel
GTSAM — implémentation du graphe de facteurs et optimisation (LevenbergMarquardtOptimizer)
Python
Structure du dépôt
├── main.py                  # Point d'entrée principal
├── fusion/                  # Implémentations des différents graphes de facteurs (FGO)
├── control_scripts/         # Scripts de génération de trajectoires (carré, triangle, vertical, rotation)
├── data_read/                # Lecture des données capteurs (GPS, IMU, odométrie, optical flow)
├── plot_scripts/             # Génération des graphes de résultats (trajectoires, RMSE)
├── vol_reel/                 # Traitement des données de vol réel (hors simulation)
├── csv_files/                 # Données capteurs enregistrées
├── images/                   # Figures et résultats visuels
├── utils/                     # Fonctions utilitaires
└── dashboard.html             # Tableau de bord de visualisation
Résultats

L'algorithme FGO couplant GPS et IMU réduit significativement l'erreur de position (RMSE) par rapport au GPS brut, sur plusieurs types de trajectoires testées (vol vertical, triangle). Le temps de calcul de l'optimisation reste faible (de l'ordre de 0.05 à 0.13 s), compatible avec un usage temps réel.

Trajectoire	RMSE GPS brut	RMSE FGO (GPS+IMU)
Vertical (X)	0.74 m	0.22 m
Vertical (Y)	0.94 m	0.23 m
Vertical (Z)	1.10 m	0.14 m
Triangle (X)	0.64 m	0.07 m
Triangle (Y)	0.97 m	0.16 m
Triangle (Z)	1.02 m	0.15 m
Volet matériel

En parallèle de la partie logicielle, une partie du stage a été consacrée à l'assemblage matériel d'un drone : intégration et câblage du contrôleur de vol SpeedeeBee, de l'ESC (Electronic Speed Controller) et de l'ordinateur de bord (Raspberry Pi 5), avec réalisation des premiers tests moteurs. L'assemblage n'a pas pu être finalisé dans le temps imparti du stage.

Perspectives
Intégration du capteur Optical Flow pour une fusion à trois capteurs
Extension vers l'odométrie visuelle-inertielle (caméra + IMU)
Validation de l'approche sur des données de vol réelles
Auteur

William Berenger Kombassere — Stage Erasmus, encadré par Florin Stoican, Daniel IOAN, Radu Cioaca, laboratoire RE.PLAN, Université Polytechnique de Bucarest (2025-2026).
