import numpy as np
import config
import os
import helper
import argparse

# Outil pour convertir le dataset au format théo vers le format mvsnet

# Format théo:
#  [tag]rgb[n].png for an image of the set [tag] at position [n] in the array
#  [tag]depth[n].png for a disparity map of the set [tag] at position [n] in the array

# DEPLACE (et pas copie, permet d'accélerer le processus)
# Les fichiers de ce tag dans un nouveau répertoire au format mvsnet
# avec le même nom. Ce nouveau répertoire contiendra les dossiers "images/" et "cams/"
# dataset_dir:
#   Contient les fichiers .png du dataset de THéo
# tag:
#   Le tag de l'ensemble des images du dataset à convertir
def convert_theo_to_mvsnet(dataset_dir: str, tag: str) -> None:
    tag_dir = os.path.join(dataset_dir, tag)
    images_dir = os.path.join(tag_dir, "images")
    
    print(f"Populating '{tag_dir}'...")

    os.makedirs(tag_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    for camera_id in range(config.num_cameras):

        # zyhb8kn5tpqei3x23o41sfrgb0.png
        old_image_file = os.path.join(dataset_dir, f"{tag}rgb{camera_id}.png")

        # 00000000.jpg
        # TODO: convert to jpg
        new_image_file = os.path.join(images_dir, f"{camera_id:08}.png")
        os.replace(old_image_file, new_image_file)
        
def main():
    parser = argparse.ArgumentParser(description="Convertit un répertoire de données au format Théo vers le format MVSNet.")
    parser.add_argument("source_dir", type=str, help='Le répertoire source qui contient les images "[tag](rgb|depth)[0-15].png"')
    parser.add_argument("--tag", type=str, help="Convertit uniquement le dataset pour ce tag.")
    args = parser.parse_args()

    if args.tag is not None:
        convert_theo_to_mvsnet(args.source_dir, args.tag)

if __name__ == "__main__":
    main()
    
    r'''
    dataset_dir = r'C:\Users\raphael\Documents\data\short\V6_mixed'
    tag = "zxd0ddlaixrtyrmh0zn5ij"
    convert_theo_to_mvsnet(dataset_dir, tag)
    '''