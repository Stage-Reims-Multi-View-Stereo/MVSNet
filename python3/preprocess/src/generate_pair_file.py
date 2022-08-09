# MVSNet a besoin de savoir, pour chaque caméra, les K caméras les plus "proches" avec un score donné à chacun.
# Pour N caméras, on calcule le score de chaque caméra avec les N-1 autres caméras.
# Les K caméras les plus proches choisies sont celles qui ont le plus haut score.

# Ici, comme les caméras sont en grille, on peut utiliser une fonction de coût qui est juste la distance entre chaque caméra.
# Les caméras choisies seront alors les caméras voisines.

import src.config as config
import src.helper as helper
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# pos_cameras: Tableau 2D de taille (N, 2). Contient les positions des N caméras.
#       Chaque ligne n° i donne la position (x, y) de la caméra n° i.
#
# return:
#       Un tableau 2D de taille (N, N). Pour chaque indice (i, j) donne le score de la caméra n° i avec la caméra n° j.
#
#
def compute_scores(pos_cameras: np.ndarray) -> np.ndarray:

    num_cameras = pos_cameras.shape[0]
    scores = np.empty((num_cameras, num_cameras))

    for i in range(num_cameras):
        for j in range(num_cameras):
            scores[i, j] = score_function(pos_cameras[i], pos_cameras[j])

    return scores


# La fonction de score est simplement l'opposé de la distance entre les deux caméras (pour que deux caméras proches aient un score élevé).
# Sauf si les caméras sont les mêmes (= ont la même position), dans ce cas on donne comme score -Infinity car on ne veut que les meilleurs voisins (et pas que les caméras s'associent avec elles-mêmes).
def score_function(camera_1, camera_2) -> float:

    distance = np.linalg.norm(camera_1 - camera_2)
    
    if distance == 0:
        return -np.inf
    else:
        # return -distance
        return 1 / distance


# pos_cameras: Tableau 2D de taille (N, 2). Contient les positions des N caméras.
#       Chaque ligne n° i donne la position (x, y) de la caméra n° i.
#
# num_bests:
#       Nombre de meilleurs voisins à extraire
#
# return: Un 2-uple (best_scores, best_indices).
#           `best_scores` et `best_indices` sont des tableaux 2D de taille (N, num_bests). Pour chaque indice (i, j), donne :
#               - Pour best_scores: Le (j+1)-ième meilleur score.
#               - Pour best_indices: La (j+1)-ième meilleure caméra associée au score.
#
def compute_best_scores(pos_cameras: np.ndarray, num_bests: int) -> np.ndarray:

    num_cameras = pos_cameras.shape[0]
    scores = compute_scores(pos_cameras)

    best_scores = np.empty((num_cameras, num_bests))
    best_indices = np.empty((num_cameras, num_bests), dtype=int)

    best_indices = np.argsort(-scores)[:, :num_bests] # minus to sort in descending order
    best_scores = -np.sort(-scores)[:, :num_bests]
    
    return (best_scores, best_indices)

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


# Affiche l'arrangement des caméras sous forme de graphique pour chaque caméra
#
# L'ordre Y affiché dans le tableau est le même que l'ordre quand il est lu dans le code source.
# Par ex: [[1, 2]
#           3, 4]] sera affiché de la même façon sur le graphe
# (effectué grâce à imshow(origin='upper'))
#
# Paramètres:
#   pos_cameras: Valeur de retour de arange_cameras()
#   best_scores, best_indices: Valeur de retour de compute_best_scores()
def plot_arangement(pos_cameras: np.ndarray, best_scores: np.ndarray, best_indices: np.ndarray) -> None:
    
    num_cameras = pos_cameras.shape[0]
    sqrt_num_cameras = int(np.sqrt(num_cameras))

    for camera_index in range(num_cameras):
        
        COLOR_CURRENT = 2
        COLOR_NEIGHBOUR = 1

        camera_pos = tuple(pos_cameras[camera_index, 0:2])

        arrangement_image = np.zeros((sqrt_num_cameras, sqrt_num_cameras))

        # highlight strongly current camera
        arrangement_image[camera_pos] = COLOR_CURRENT

        # higlight neighbouring cameras
        pos_best_neighbours = tuple(np.transpose(pos_cameras[best_indices[camera_index, :]]))
        arrangement_image[pos_best_neighbours] = COLOR_NEIGHBOUR

        print(":== Plot ==:")
        print(f"Camera: index {camera_index}, pos. {camera_pos}")
        print(f"Neighbours:\n{pos_best_neighbours}")
        print(f"Arrangement:\n{arrangement_image}")

        plt.imshow(arrangement_image)

        ## Texte
        for i in range(num_cameras):
            row, col = pos_cameras[i, 0:2] # Order in code
            x, y = col, row # Order X/Y is flipped
            label = str(i)
            plt.text(x, y, label, ha='center', va='center')
        ## Fin Texte

        plt.colorbar()
        plt.show()


# Ecrit un fichier au format pair.txt attendu par MVSNet: https://github.com/YoYo000/MVSNet#file-formats
# Les paramètres sont la sortie de compute_best_scores()
def save_as_mvsnet_pairs_file(best_scores, best_indices, output_file_name: str, min_score: float) -> None:
    
    print(f"Generating {output_file_name}...")

    num_cameras = best_indices.shape[0]
    num_neighbours = best_indices.shape[1]
    lines = []

    # TOTAL_IMAGE_NUM
    lines.append(str(num_cameras))

    for camera_index in range(num_cameras):
        # IMAGE_IDi
        lines.append(str(camera_index))

        # 10 ID0 SCORE0 ID1 SCORE1 ...
        line_as_numbers = []
        
        # Limit neighbours by score
        # (score is 1 / distance)
        current_num_neighbours = np.count_nonzero(
            best_scores[camera_index] >= min_score)
        
        print(f"Num. neighbours for camera n° {camera_index}: {current_num_neighbours}")

        line_as_numbers.append(str(current_num_neighbours))

        for neighbour_index in range(current_num_neighbours):
            line_as_numbers.append(str(best_indices[camera_index, neighbour_index]))
            line_as_numbers.append(str(best_scores[camera_index, neighbour_index]))

        lines.append(" ".join(line_as_numbers))

    helper.write_lines(output_file_name, lines)


def main(pair_path: str, max_neighbours: int, min_neighbours_score: float, pos_cameras):
    
    best_scores, best_indices = compute_best_scores(pos_cameras, max_neighbours)


    print("Position des caméras:")
    print(pos_cameras)

    print(f'Extrait les max. {max_neighbours} meilleures voisins.')
    print('pos_cameras:\n', pos_cameras)

    print('best_scores:\n', best_scores)
    print('best_indices:\n', best_indices)

    os.makedirs(os.path.abspath(os.path.join(pair_path, os.pardir)), exist_ok=True)
    save_as_mvsnet_pairs_file(best_scores, best_indices, pair_path, min_neighbours_score)


if __name__ == '__main__':
    
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    
    main()
