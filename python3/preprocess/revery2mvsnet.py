#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import shutil
import os

PROGRAM_DESCRIPTION = """
Convertit un répertoire au format ReVeRy vers le format MVSNet,
pour pouvoir l'utiliser avec ce dernier.
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
                sub_dir = "images"
            else:
                sub_dir = "rendered_depth_maps"

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
                    

        dataset_id += 1


def main():
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input",        type=str,             help='Le répertoire source qui contient les images "[tag](rgb|depth)[0-15].png"', required=True)
    parser.add_argument("-o", "--output",       type=str,             help="Le répertoire à créer.", required=True)
    parser.add_argument("-n", "--max_datasets", type=int, default=-1, help='Limite le nombre de datasets à convertir (-1 pour illimité).')
    args = parser.parse_args()

    convert_validation_set(args.input, args.output, args.max_datasets)


if __name__ == "__main__":
    main()
