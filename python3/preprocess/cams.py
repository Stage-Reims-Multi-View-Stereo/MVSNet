#!/usr/bin/env python3

import argparse 
import os
import src.config as config
import src.generate_cams_dir as generate_cams_dir
import src.generate_pair_file as generate_pair_file
import numpy as np
import json

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description="Générer le répertoire de layout de caméras.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-o", "--cams_dir",          type=str,    default=   None    ,  help="Répertoire à créer et où générer pair.txt ainsi que les fichiers 000000[0-15]_cam.txt.", required=True)
parser.add_argument("-n", "--max_neighbours",    type=int,    default=   10      ,  help="Nombre maximum de voisins pour chaque caméra.")
parser.add_argument("-m", "--max_distance",      type=float,  default=   1.0     ,  help="Distance max. d'un voisin. Pour sélectionner uniquement les 4 (ou moins pour les bords) voisins adjacents, mettre 1.0.")
parser.add_argument("-b", "--baseline",          type=float,  default=   0.3     ,  help="Distance entre les caméras.")
parser.add_argument("-f", "--fovy",              type=float,  default=   60      ,  help="FOV Vertical, en degrés.")
parser.add_argument("-W", "--image_width",       type=int,    default=   1920    ,  help="Largeur des images.")
parser.add_argument("-H", "--image_height",      type=int,    default=   1080    ,  help="Hauteur des images.")
parser.add_argument("-d", "--depth_min",         type=float,  default=   1.0     ,  help="Profondeur maximum.")
parser.add_argument("-i", "--depth_interval",    type=float,  default=   0.5     ,  help="Intervalle entre deux valeurs de profondeurs (définit la précision).")
args = parser.parse_args()


indices_cameras = config.indices_cameras
max_neighbours = args.max_neighbours
max_distance = args.max_distance
pos_cameras = generate_cams_dir.arange_cameras(indices_cameras)
cams_dir = args.cams_dir
image_width = args.image_width
image_height = args.image_height
fovy = np.radians(args.fovy)
depth_min = args.depth_min
depth_interval = args.depth_interval
baseline = args.baseline
pair_path = os.path.join(cams_dir, 'pair.txt')

os.makedirs(cams_dir, exist_ok=True)    

intrinsic_matrix = generate_cams_dir.build_intrinsic_matrix(image_width, image_height, fovy)
generate_cams_dir.build_cams_dir(cams_dir, intrinsic_matrix, pos_cameras, depth_min, depth_interval, baseline)
generate_pair_file.main(pair_path, max_neighbours, max_distance, pos_cameras)

disparity_to_depth_numerator = (
    (baseline * image_height) /
    (2 * np.tan(fovy / 2))
)

generation_settings = {}
generation_settings["_comment"] = "Ne pas modifier. Ce fichier contient les paramètres avec lesquels ont été générés les fichiers de caméra de ce dossier. 'fovy' est en radians."
generation_settings["fovy"] = fovy
generation_settings["depth_min"] = depth_min
generation_settings["depth_interval"] = depth_interval
generation_settings["baseline"] = baseline
generation_settings["image_width"] = image_width
generation_settings["image_height"] = image_height
generation_settings["disparity_numerator"] = disparity_to_depth_numerator

json.dump(
    generation_settings,
    open(os.path.join(cams_dir, "generation_settings.json"), "w"),
    indent=4, sort_keys=True,
    ensure_ascii=False)
