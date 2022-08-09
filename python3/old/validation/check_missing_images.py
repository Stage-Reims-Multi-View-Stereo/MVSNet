#!/usr/bin/env python3

import argparse
from genericpath import isfile
from pathlib import Path
import re
import shutil
import os

PROGRAM_DESCRIPTION = """
Vérifie s'il manque certaines images dans le dataset.
"""
def read_lines(filename):
    with open(filename) as f:
        return f.read().splitlines()

def check_file_exists(path: str):
    if not os.path.isfile(path):
        print(f"Missing file: '{path}'")


def warn_missing_dir(path: str):
    print(f"Missing directory: '{path}'")


def check_missing_images(src_dir: str):
    scenes_uuids = read_lines(os.path.join(
        src_dir, "all_list.txt"))

    for uuid in scenes_uuids:
        scene_dir = os.path.join(src_dir, uuid)

        if os.path.exists(scene_dir):
            depth_dir = os.path.join(scene_dir, "rendered_depth_maps")
            color_dir = os.path.join(scene_dir, "images")

            if os.path.exists(depth_dir):
                for i in range(16):
                    file = os.path.join(depth_dir, str(i).zfill(8) + ".png")
                    check_file_exists(file)
            else:
                warn_missing_dir(depth_dir)

            if os.path.exists(color_dir):
                for i in range(16):
                    file = os.path.join(depth_dir, str(i).zfill(8) + ".pfm")
                    check_file_exists(file)
                pass
            else:
                warn_missing_dir(color_dir)

        else:
            warn_missing_dir(scene_dir)


def main():
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION)
    parser.add_argument("-i", "--dataset_dir", type=str,
                        help='Le répertoire du dataset"', required=True)
    args = parser.parse_args()

    check_missing_images(args.dataset_dir)


if __name__ == "__main__":
    main()
