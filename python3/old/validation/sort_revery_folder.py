#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import shutil
import os
import mvsnet_conventions

PROGRAM_DESCRIPTION = """
Convertit un répertoire au format Théo vers le format BlendedMVS.
Cela permet de faire croire à MVSNet que le jeu de données est BlendedMVS et
par conséquent l'utiliser pour entraîner MVSNet.
Créer aussi des symlinks '<uuid>/cams' => dst/cams' dans chaque répertoire de scène, comme toutes les caméras ont
sont du même modèle, et cela économise de la place et du temps de copie.
"""


def ensure_parent_exists(path):
    parent_path = Path(path).parent
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)


def convert_validation_set(src_dir: str, dst_dir: str, max_datasets: int = -1):

    # Group 1: Base directory
    # Group 2: UUID
    # Group 3: rgb | depth
    # Group 4: Image index
    pattern = re.compile("(.+)/(\w+)(rgb|depth)(\d+).png")

    dataset_id = 0

    already_created_cams_dir = set()

    for png_path in sorted(Path(src_dir).rglob("*.png")):

        if max_datasets != -1 and dataset_id >= max_datasets:
            print(f"Stopped after having converted {max_datasets} datasets.")
            break

        png_abspath = str(png_path)  # Path -> str

        match = pattern.fullmatch(png_abspath)

        if match:
            base_dir = match.group(1)
            uuid = match.group(2)
            image_type = match.group(3)
            image_id = match.group(4).rjust(8, '0')  # fill with zeros

            if image_type == "rgb":
                sub_dir = mvsnet_conventions.DIR_RGB
            else:
                sub_dir = mvsnet_conventions.DIR_DEPTH

            scene_dir = os.path.join(dst_dir, uuid)
            new_png_path = os.path.join(scene_dir, sub_dir, image_id + ".png")
            ensure_parent_exists(new_png_path)

            info = ("'" + png_abspath + "' -> '" + new_png_path + "'")

            if os.path.exists(new_png_path):
                print(info, "(already exist, skipped)")
            else:
                print(info)
                shutil.copy(png_path, new_png_path)

            if not uuid in already_created_cams_dir:

                # filter to accelerate the process (but not critical)
                # only file copy is really critical
                already_created_cams_dir.add(uuid)

                # we create a broken symlink if ../cam doesn't exist,
                # that doesn't matter we can create it later
                src = "../" + mvsnet_conventions.DIR_CAMS
                dst = os.path.join(scene_dir, mvsnet_conventions.DIR_CAMS)

                # os.path.exist(dst) returns false if dst is broken symlink
                if os.path.exists(dst) or os.path.islink(dst):
                    print(
                        f"Path '{dst}' already exist, symlink creation skipped")
                else:
                    os.symlink(src, dst, target_is_directory=True)
                    print(f"Created symlink '{dst}' -> '{src}'")

                dst = os.path.join(scene_dir, mvsnet_conventions.PAIR_FILE)
                src = "../" + mvsnet_conventions.PAIR_FILE
                # Nécessaire pour la phase de test (mais pas d'entraînement)
                # Autant l'ajouter tout de suite
                if os.path.exists(dst) or os.path.islink(dst):
                    print(f"Path '{dst}' already exist, symlink creation skipped")
                else:
                    os.symlink(src, dst, target_is_directory=True)
                    print(f"Created symlink '{dst}' -> '{src}'")
                    

        dataset_id += 1


def main():
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION)
    parser.add_argument(
        "src", type=str, help='Le répertoire source qui contient les images "[tag](rgb|depth)[0-15].png"')
    parser.add_argument("dst", type=str, help="Le répertoire à créer.")
    parser.add_argument("-n", metavar="max_datasets", type=int,
                        help='Limite le nombre de datasets à convertir (-1 pour illimité).')
    args = parser.parse_args()

    convert_validation_set(args.src, args.dst, args.n or -1)


if __name__ == "__main__":
    main()
