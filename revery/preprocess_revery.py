# -*- coding: utf-8 -*-

from __future__ import print_function
from genericpath import isfile

import os
import time
import glob
import random
import math
import re
import sys

import cv2
import numpy as np
import tensorflow as tf
import scipy.io
import urllib
from tensorflow.python.lib.io import file_io
FLAGS = tf.app.flags.FLAGS

def gen_revery_path(revery_data_folder, mode='training'):
    """
    Base copiée depuis mvsnet/preprocess.py:gen_blendedmvs_path().
    mode: 'training', 'validation', ou 'test'.
    
    On a cependant besoin de faire quelques modifications:
    
    - Toutes les scènes possèdent le même layout de caméras,
      et les caméras sont toutes identiques, donc on cherche les chemins de caméra à
      la racine du dataset et non pas dans chaque scène.
    
    - Changement des noms des fichiers d'entraînement pour ne plus faire référence à
      BlendedMVS.
    
    - Si c'est pour les tests, revery_data_folder ne doit pas pointer à la racine
      du répertoire revery mais directement dans le sous-répertoire de la scène voulue.
    
    Les modifications opérées sont commentées par [REVERY].
    """

    # read data list
    # [REVERY] modification des noms des .txt
    if mode == 'training':
        proj_list = open(os.path.join(revery_data_folder, 'training_list.txt')).read().splitlines()
    elif mode == 'validation':
        proj_list = open(os.path.join(revery_data_folder, 'validation_list.txt')).read().splitlines()
    elif mode == 'testing':
        # [REVERY] Ajout d'une option pour tester
        # Le fichier 'testing_list.txt' existe mais n'est pas utilisé ici
        proj_list = [os.path.basename(os.path.abspath(revery_data_folder))] # abspath() in case of trailing /
        revery_data_folder = os.path.join(revery_data_folder, os.pardir)
    else:
        raise ValueError("Unknown mode:", mode)
    
    # parse all data
    mvs_input_list = []
    for data_name in proj_list:

        dataset_folder = os.path.join(revery_data_folder, data_name)

        # read cluster
        # [REVERY] 'pair.txt' commun à toutes les scènes
        cluster_path = os.path.join(revery_data_folder, 'cams', 'pair.txt')
        cluster_lines = open(cluster_path).read().splitlines()
        image_num = int(cluster_lines[0])

        # get per-image info
        for idx in range(0, image_num):

            ref_idx = int(cluster_lines[2 * idx + 1])
            cluster_info =  cluster_lines[2 * idx + 2].split()
            total_view_num = int(cluster_info[0])
            if total_view_num < FLAGS.view_num - 1:
                continue
            paths = []
            
            # [REVERY] changer 'blended_images' par 'images'
            # [REVERY] changer 'jpg' par 'png'
            # [REVERY] retirer '_masked'
            ref_image_path = os.path.join(dataset_folder, 'images', '%08d.png' % ref_idx)
            ref_depth_path = os.path.join(dataset_folder, 'rendered_depth_maps', '%08d.pfm' % ref_idx)
            # [REVERY] 'cams/' commun à toutes les scènes
            ref_cam_path = os.path.join(revery_data_folder, 'cams', '%08d_cam.txt' % ref_idx)
            paths.append(ref_image_path)
            paths.append(ref_cam_path)

            for cidx in range(0, FLAGS.view_num - 1):
                view_idx = int(cluster_info[2 * cidx + 1])
                # [REVERY] changer 'blended_images' par 'images'
                # [REVERY] changer 'jpg' par 'png'
                # [REVERY] retirer '_masked'
                view_image_path = os.path.join(dataset_folder, 'images', '%08d.png' % view_idx)
                # [REVERY] 'cams/' commun à toutes les scènes
                view_cam_path = os.path.join(revery_data_folder, 'cams', '%08d_cam.txt' % view_idx)
                paths.append(view_image_path)
                paths.append(view_cam_path)
            
            # [REVERY] Permet d'utiliser la même fonction pour les tests et pour le train/validation
            # Les tests n'attendent pas un chemin pour la depth map
            if mode != 'testing':
                paths.append(ref_depth_path)
            
            # [REVERY] Ajout d'un check que les fichiers existent (debugging)
            for path in paths:
                if not os.path.isfile(path):
                    raise ValueError("File not found: " + path)

            mvs_input_list.append(paths)

    return mvs_input_list