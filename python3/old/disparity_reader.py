# Basé sur le fichier disparity_reader.py du dataset Revery

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

# Comme les fichiers .fpm prennent beaucoup de places,
# il a été utilisé des fichiers .png pour prendre moins de place dans lesquels
# la disparité est packée suivant la formule dans le code
# return:
#   La carte de disparité dans un tableau 2D
def read_disparity_map(path: str) -> np.ndarray:
    tmp = cv2.imread(path, -1)

    try:
        # print(path)
        b, g, r, a = cv2.split(tmp)

    except:
        print(path)
        exit(1)

    res = r + g / 256.0 + b / (256.0 * 256.0) + a / (256.0 * 256.0 * 256.0)

    return 32 * res