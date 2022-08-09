#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# [REVERY] ne pas utiliser tensorflow car le temps d'import est trop long, si le programme est lancé beaucoup de fois pour chaque image du dataset
# from tensorflow.python.lib.io import file_io
import config
import argparse
import cv2
import sys
import re
import argparse
import os

def read_pfm(filename):
    '''Source: MVSNet/mvsnet/preprocess.py'''
    
    file = open(file, 'rb')
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = file.readline().decode('UTF-8').rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).decode('UTF-8').rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data

def write_pfm(filename, image, scale=1):
    '''Source: MVSNet/mvsnet/preprocess.py. scale est toujours laissé à 1 dans MVSNet.
    On doit utiliser le même parser que MVSNet pour être sûr que le format de fichier est bon,
    car il existe d'autres parsers sur internet mais qui flippent l'image en X ou en Y.
    
    Parameters:
        scale: Pour le header PFM. Pas utilisé par MVSNet, donc n'importe quelle valeur non-zéro fonctionne.
    '''
    
    file = open(filename, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    # [REVERY] deprecation warning: use tobytes() instead of tostring()
    image_string = image.tobytes()
    file.write(image_string)

    file.close()

def read_disparity_map(path: str) -> np.ndarray:
    """
    Basé sur le fichier disparity_reader.py du dataset Revery
    Comme les fichiers .fpm prennent beaucoup de places,
    il a été utilisé des fichiers .png pour prendre moins de place dans lesquels
    la disparité est packée suivant la formule dans le code
    
    Returns:
        La carte de disparité dans un tableau 2D
    """
    
    tmp = cv2.imread(path, -1)

    try:
        # print(path)
        b, g, r, a = cv2.split(tmp)

    except:
        print(path)
        exit(1)

    res = r + g / 256.0 + b / (256.0 * 256.0) + a / (256.0 * 256.0 * 256.0)

    return 32 * res

# Affiche la carte de disparité après conversion depuis le .png du dataset
# ET la carte de profondeur correspondante
def plot_disparity_and_depth_maps(disparity_map: np.ndarray, depth_map: np.ndarray, title: str) -> None:

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(title)

    # divider permet d'afficher une colorbar par ax,
    # normalement on ne peut afficher qu'une colorbar par figure

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im1 = ax1.imshow(disparity_map, cmap="gray")
    ax1.set_title("Disparity map")
    fig.colorbar(im1, cax=cax, orientation='vertical')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im2 = ax2.imshow(depth_map, cmap="gray")
    ax2.set_title("Depth map")
    fig.colorbar(im2, cax=cax, orientation='vertical')

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Afficher une carte de disparité et la profondeur associée.")
    parser.add_argument("-i", "--disparity_file", type=str,
                        help="Le fichier de disparité packé.")
    parser.add_argument("-o", "--save_pfm", dest="pfm_output_file", type=str,
                        help="Sauvegarder la depth map de sorti vers un fichier .pfm")
    parser.add_argument("-c", "--disp2depth", dest="disparity", type=float,
                        help="Convertir une valeur de disparité en depth dans la console")
    parser.add_argument("-p", "--plot", action=argparse.BooleanOptionalAction,
                        help="Afficher les cartes de profondeur et de disparité dans une fenêtre")
    parser.add_argument("-b", "--baseline", type=float, help="Distance entre les caméras", required=True)
    parser.add_argument("-f", "--fovy", type=float, help="FOVY, en degres (si c'est en radians, spécifier l'option -r).", required=True)
    parser.add_argument("-H", "--image-height", type=int, help="Hauteur de l'image, en pixels.")
    parser.add_argument("-r", "--radians", help="Passer le FOVY en radians et non pas en degrés", default=False, action='store_true')
    
    args = parser.parse_args()
    baseline = args.baseline
    image_height = float(args.image_height)
    fovy = args.fovy if args.radians else np.radians(args.fovy)
    
    # Convertit une disparité (en pixels) en profondeur (en mètre)
    # disparity: fonctionne à la fois avec des scalaires (floats) et des tableaux numpy
    disparity_to_depth_numerator = (
        (baseline * image_height) /
        (2 * np.tan(fovy / 2))
    )

    if args.disparity is not None:
        disparity = args.disparity
        depth = disparity_to_depth_numerator / disparity
        print(f"disparity={disparity}")
        print(f"depth={depth}")

    if args.disparity_file is not None:
        disparity_file_name = args.disparity_file

        disparity_map = read_disparity_map(
            disparity_file_name)
        depth_map = disparity_to_depth_numerator / disparity_map

        if args.plot:
            plot_disparity_and_depth_maps(
                disparity_map, depth_map, disparity_file_name)

        # save pfm if asked
        if args.pfm_output_file is not None:
            write_pfm(args.pfm_output_file, depth_map.astype(np.float32))


if __name__ == "__main__":
    main()
