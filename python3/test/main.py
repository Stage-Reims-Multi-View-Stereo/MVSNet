#!/usr/bin/env python3

import cv2
import argparse
import os
import glob
import sys
import numpy as np
import skimage.metrics
import matplotlib.pyplot as plt

def print_error(msg):
    sys.stderr.write(msg + "\n")


def remove_extension(path):
    return os.path.splitext(path)[0]


def build_image_diff(image1, image2):
    # Compute SSIM between two images
    score, diff = skimage.metrics.structural_similarity(image1, image2, full=True)

    # The diff image contains the actual image differences between the two images
    
    return score, diff


def compute_stats_for_scene(scene, imshow=False):
    """
    Calcule des variables statistiques pour la scène donnée.
    
    Parameters:
        scene (list): La liste des scène sous une forme de liste de paires (A, B) où A est le chemin vers
                      la vérité terrain et B et le chemin vers l'image inférée par le réseau de neurones.
    """
    
    # Ne charge pas toutes les images d'un coup, mais
    # en itérant, sinon cela consommerait trop de mémoire (environ 500Mo pour x16 images Full HD)

    ret = {
        "rmse": [],
        "ssim": []
    }
    
    for ground_truth_path, inferred_path in scene:
        ground_truth_image = cv2.imread(ground_truth_path)
        inferred_image = cv2.imread(inferred_path)
        
        # Most of the time, inferred depth is samaller than original depth image
        ground_truth_image = cv2.resize(ground_truth_image, dsize=inferred_image.shape[::-1], interpolation=cv2.INTER_LINEAR)
        rmse = np.sqrt(np.mean((ground_truth_image - inferred_image) ** 2))
        (ssim, diff_image) = build_image_diff(ground_truth_image, inferred_image)
        
        if imshow:
            plt.imshow(ground_truth_image, cmap='rainbow')
            plt.show()
            
            plt.imshow(inferred_image, cmap='rainbow')
            plt.show()
            
            plt.imshow(diff_image, cmap='gist_gray')
            plt.show()
        
        ret["rmse"].append(rmse)
        ret["ssim"].append(ssim)

    return ret
    

def plot_stats(stats):
    """
    Args:
        stats (dict): Dictionnaire des valeurs, pour chaque entrée contient une liste des mesures pour chaque image (rmse, ssim...) 
    """
    xs = stats["rmse"]
    ys = stats["ssim"]
    print(xs, ys)
    plt.scatter(xs, ys)
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print image paths of the depth of ground truth and inferred", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input",        type=str,             help='Le répertoire de la scène', required=True)
    parser.add_argument("--plot_images",  type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot_stats",  type=bool, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    
    scene = load_scene(args.input)
    stats = compute_stats_for_scene(scene, args.plot_images)
    
    if args.plot_stats:
        plot_stats(stats)