#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import shutil
import os
import glob
import random
import mvsnet_conventions

PROGRAM_DESCRIPTION = """
Générer les fichiers nécessaires pour le format BlendedMVS
qui sépare le jeu de données en deux parties, training et validation."""


def write_lines(output_file_name: str, lines: list) -> None:

    with open(output_file_name, 'w') as output_file:
        output_file.write("\n".join(lines))


def generate_validation_split_files(dst_dir: str, validation_percent: float, test_percent: float, scene_count: int):

    # Liste des scènes
    scenes_uuids = [os.path.basename(os.path.normpath(x))
                    for x in glob.glob(dst_dir + "/*/")]
    
    if scene_count > 0:
        scenes_uuids = scenes_uuids[:scene_count] # Limit count of scenes

    scenes_uuids = sorted(scenes_uuids)
    num_scenes = len(scenes_uuids)
    num_validation = int(num_scenes * validation_percent)
    num_test = int(num_scenes * test_percent)
    num_train = num_scenes - num_validation - num_test

    print(f"Total scenes count: {num_scenes}")
    print(
        f"Validation count {num_validation}/{num_scenes} ({(num_validation / num_scenes * 100):.2f}%)")
    print(f"Training count: {num_train}/{num_scenes} ({(num_train / num_scenes * 100):.2f}%)")
    print(f"Test count: {num_test}/{num_scenes} ({(num_test / num_scenes * 100):.2f}%)")
    print("Splitting method: random")

    random.shuffle(scenes_uuids)

    # list order: train     |   validation  | test
    # indices:    0  ...      num_train       num_train + num_validation
    
    test_start_id = num_train + num_validation

    write_lines(os.path.join(dst_dir, mvsnet_conventions.FILENAME_ALL),
                scenes_uuids)
    write_lines(os.path.join(dst_dir, mvsnet_conventions.FILENAME_TRAINING),
                scenes_uuids[:num_train])
    write_lines(os.path.join(dst_dir, mvsnet_conventions.FILENAME_VALIDATION),
                scenes_uuids[num_train:test_start_id])
    write_lines(os.path.join(dst_dir, mvsnet_conventions.FILENAME_TEST),
                scenes_uuids[test_start_id:])


def main():
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION)
    parser.add_argument(
        "dst_dir", type=str, help="Le répertoire SCENES qui contient un répertoire SCENES/<UUID> par scène.")
    parser.add_argument("validation_percent", type=float,
                        help='Pourcentage de scènes à utiliser pour la validation (entre 0.0 et 1.0).')
    parser.add_argument("test_percent", type=float,
                        help='Pourcentage de scènes à utiliser pour les tests (entre 0.0 et 1.0).')
    parser.add_argument("-n", metavar="scene_count", type=int, default=-1, help="Nombre de scènes maximum à utiliser")
    args = parser.parse_args()

    generate_validation_split_files(args.dst_dir, args.validation_percent, args.test_percent, args.n)


if __name__ == "__main__":
    main()
