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

# Nombre de plus proches voisins à extraire pour chaque caméra
num_camera_neighbours = 10

# Extrait les voisins d'une distance inférieur ou égal
# Utile pour extraire les 4 voisins adjacents (mettre 1.0)
# Le nombre de voisins peut varier par caméra
min_neighbour_score = 1.0

# Distance entre les caméras, en mètres
cameras_baseline = 0.30

# Paramètres nécessaires pour la construction des fichiers de caméra
depth_min = 1.0
depth_interval = 1.0

# Taille des images du dataset
# Constante pour toutes les images du dataset
# Les cartes de profondeur / couleurs doivent avoir la même taille
image_width = 1920
image_height = 1080

# FOV en X et en Y de la caméra
# Constante pour toutes les images du dataset
# En détermine le FOVX suivant les autres paramètres
# NOTE: si FOVY = 60° en 1920x1080, alors FOVx = 80°
camera_fovy = np.radians(60)
camera_fovx = deduce_fovx(camera_fovy, image_width, image_height)


# NOTE:
#   La valeur de la distance focale est arbitraire (surtout pour des images générées par ordinateur)
#   connaissant le FOV, la valeur (baseline / focal_length) est fixée.
#   Donc, à une constante de proportionalité près.
#   Par exemple, OpenGL à une baseline = 2. C'est utile car cela simplifie les calculs trigonométriques. 

def main():
    # Print variables when executed
    prefix = "indices_cameras:"
    print(prefix, np.array2string(indices_cameras, prefix=prefix))
    print(f"{num_camera_neighbours=}")
    print(f"{cameras_baseline=}m")
    print(f"{image_width=}px")
    print(f"{image_height=}px")
    print(f"{camera_fovy=}rad ({np.degrees(camera_fovy)}°)")
    print(f"{camera_fovx=}rad ({np.degrees(camera_fovx)}°)")

if __name__ == "__main__":
    main()
