import numpy as np

def deduce_fovx(fovy: float, image_width: float, image_height: float) -> float:

    aspect_ratio = (image_width / image_height)
    fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect_ratio)

    return fovx
    
# Disposition des caméras = layout
'''
Extrait README du dataset:
    - Positions are written /!\ down to up /!\ and left to right, i.e the camera positions are :
            
            12 13 14 15
            8   9 10 11
            4   5  6  7
            0   1  2  3
'''
indices_cameras = np.array([
    [12, 13, 14, 15],
    [8, 9, 10, 11],
    [4, 5, 6, 7],
    [0, 1, 2, 3]
], dtype=int)

# Nombre total de caméras
num_cameras = 16