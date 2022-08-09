#!/usr/bin/env python3

# MVSNet a besoin de savoir, pour chaque caméra, les K caméras les plus "proches" avec un score donné à chacun.
# Pour N caméras, on calcule le score de chaque caméra avec les N-1 autres caméras.
# Les K caméras les plus proches choisies sont celles qui ont le plus haut score.

# Ici, comme les caméras sont en grille, on peut utiliser une fonction de coût qui est juste la distance entre chaque caméra.
# Les caméras choisies seront alors les caméras voisines.

import config
import numpy as np
import matplotlib.pyplot as plt
import helper
import argparse
import os
import sys


# indices_cameras:
#       Un tableau 2D qui contient pour chaque index (i, j) l'index de la caméra à cette position.
#       En tout contient N = i * j caméras.
#       Chacun des indices dans [0;N) doit apparaître exactement une fois dans le tableau. Sinon le script le détecte et lance une erreur.
#
# return:
#       Un Tableau 2D de taille (N, 2). Contient les positions des N caméras.
#       Chaque ligne n° i donne la position (x, y) de la caméra n° i.
def arange_cameras(indices_cameras: np.ndarray) -> np.ndarray:

    num_cameras = indices_cameras.size
    pos_cameras = np.empty((num_cameras, 2), dtype=int)
    indices_not_found = set(np.arange(num_cameras))

    for ix, iy in np.ndindex(indices_cameras.shape): # itérer chaque cellule 2D

        camera_index = indices_cameras[ix, iy]
        
        if camera_index in indices_not_found:
            
            pos_cameras[camera_index, :] = [ix, iy]
            indices_not_found.remove(camera_index)

        else:
            raise ValueError(f"Invalid camera index at position ({ix}, {iy}). Check if it in range [0;num_cameras) and there are no duplicate index.")

    return pos_cameras


# Répère MONDE choisi:
#   - Origine à la caméra à row=0, col=0
#   - +X_world dans le sens +row
#   - +Y_world dans le sens +col
#   - Pas de translation en Z donc pas besoin de définir +Z_world ici.
#
# camera_row, camera_col:
#   Position de la caméra dans le tableau des caméras.
#
# return:
#   La matrice extrinsèque 4x4 de la caméra de cette position.
def build_extrinsic_matrix(camera_row: int, camera_col: int) -> np.ndarray:
    
    # Construire à partir d'une matrice identité
    extrinsic_matrix = np.identity(4)

    # Camera translation in X
    extrinsic_matrix[0, 3] = config.cameras_baseline * camera_row

    # Camera translation in Y
    extrinsic_matrix[1, 3] = config.cameras_baseline * camera_col

    return extrinsic_matrix

def deduce_focal_length(captor_height: float, fovy: float) -> float:
    focal_length = captor_height / (2 * np.tan(fovy / 2))
    return focal_length

def deduce_captor_height(focal_length: float, fovy: float) -> float:
    captor_height = focal_length * (2 * np.tan(fovy / 2))
    return captor_height

# return:
#   La matrice intrinsèque 3x3 partagée entre toutes les caméras.
def build_intrinsic_matrix() -> np.ndarray:

    # Construire à partir d'une matrice identité
    intrinsic_matrix = np.identity(3)

    # Question: La distance focale en m. ou en px.?
    # Après avoir essayé avec les m., on essaye en px., peut être que les résultats seront meilleurs
    captor_height = config.image_height
    captor_width = config.image_width
    
    '''
    # On définit ARBITRAIREMENT la taille du capteur (tant que le ratio est respecté)
    captor_height = 1
    captor_width = captor_height * (config.image_width / config.image_height)
    '''

    # On peut déduire la distance focale d'après les autres paramètres
    focal_length = deduce_focal_length(captor_height, config.camera_fovy)

    '''
    # En définissant d'abord focal_length
    focal_length = 0.035
    captor_height = deduce_captor_height(focal_length, config.camera_fovy)
    captor_width = captor_height * (config.image_width / config.image_height)
    '''

    print("*** Intrinsic matrix build parameters ***")
    print(f"{focal_length=}")
    print(f"{captor_width=}")
    print(f"{captor_height=}")
    
    # ce n'est PAS l'aspect ratio (l'aspect ratio est w/h)
    a = config.image_width / captor_width

    # Calculs trigonométriques
    # Voir: https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec
    
    # Facteur d'aggrandissement intrinsèque (Sx, Sy)
    intrinsic_matrix[0, 0] = a * focal_length
    intrinsic_matrix[1, 1] = a * focal_length

    # Translation intrinsèque (Tx, Ty)
    intrinsic_matrix[0, 2] = config.image_width / 2
    intrinsic_matrix[1, 2] = config.image_height / 2

    return intrinsic_matrix

# Chaque dossier de tag doit contenir un répertoire 'cams'
# Pour notre dataset, le répertoire est toujours le même car
# les caméras sont toujours dans la même configuration.
# Cette fonction permet de générer le répertoire dans une destination choisie.
# Il faudra ensuite le copier (ou générer des symlinks) à la main dans chacun des répertoires de tag
# Eviter d'appeler cette fonction pour générer chaque répertoire pour chaque tag car c'est relativement long,
# copier ou alors encore mieux utiliser des symlinks à la place.
#
# output_directory:
#       Répertoires à créer.
#       Les fichiers créés à l'intérieur seront:
#           - output_directory/00000000_cam.txt
#           - ...
#           - output_directory/00000015_cam.txt
#
# intrinsic:
#       Matrice 3x3 des paramètres intrinsèques de la caméra.
#       La matrix sera écrite à l'identique dans le fichier de caméras.
#       Voir les exemples de eth3d (eth3d/delivery_area/cams/00000001_cam.txt)
#
def build_cams_dir(output_dir: str, intrinsic_matrix: np.ndarray, pos_cameras: np.ndarray, depth_min: float, depth_interval: float) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Format du txt:
    '''
    extrinsic
    E00 E01 E02 E03
    E10 E11 E12 E13
    E20 E21 E22 E23
    E30 E31 E32 E33

    intrinsic
    K00 K01 K02
    K10 K11 K12
    K20 K21 K22

    DEPTH_MIN DEPTH_INTERVAL (DEPTH_NUM DEPTH_MAX) 
    '''

    os.makedirs(output_dir, exist_ok=True)

    for camera_id in range(config.num_cameras):
        
        camera_row, camera_col = pos_cameras[camera_id, 0:2]
        extrinsic_matrix = build_extrinsic_matrix(camera_row, camera_col)

        # Build le fichier de caméra
        
        lines = []
        
        lines.append("extrinsic")
        lines.append(helper.array2string_nobrackets(extrinsic_matrix))
        lines.append("intrinsic")
        lines.append(helper.array2string_nobrackets(intrinsic_matrix))
        lines.append(f"{depth_min} {depth_interval}")

        helper.write_lines(f"{output_dir}/{camera_id:08}_cam.txt", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Générer le répertoire cams/ pour MVSNet.")
    parser.add_argument("output_cam_dir", type=str, help="Créer le répertoire des fichiers de caméra 'cams' dans ce répertoire et y générer les fichiers 000000[00-15]_cam.txt")
    args = parser.parse_args()
    
    indices_cameras = config.indices_cameras
    
    num_bests = config.num_camera_neighbours
    pos_cameras = arange_cameras(indices_cameras)

    cams_dir = os.path.join(args.output_cam_dir, "cams")

    os.makedirs(cams_dir, exist_ok=True)    

    intrinsic_matrix = build_intrinsic_matrix()
    build_cams_dir(cams_dir, intrinsic_matrix, pos_cameras, config.depth_min, config.depth_interval)


if __name__ == '__main__':
    
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    
    main()
