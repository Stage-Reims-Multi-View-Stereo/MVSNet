#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import shutil
import os
import glob
import random

PROGRAM_DESCRIPTION = """
Sépare le dataset en une partie d'entraînement,
une de validation et une de testing en générant des fichiers textes
qui liste les scènes correspondantes à chaque partie.
Attention: Le répertoire ne doit contenir QUE des répertoires de scènes.
Chaque répertoire est considéré comme un scène."""


def write_lines(output_file_name: str, lines: list) -> None:
    with open(output_file_name, 'w') as output_file:
        output_file.write("\n".join(lines))


def generate_validation_split_files(data_dir: str, output_dir: str, validation_percent: float, test_percent: float, scene_count: int):

    # Liste des scènes
    scenes_uuids = [os.path.basename(os.path.normpath(x))
                    for x in glob.glob(data_dir + "/*/")]
    
    # On retire le répertoire 'cams' s'il a été généré avant
    scenes_uuids = list(filter(lambda path: path != 'cams', scenes_uuids))
   
    if len(scenes_uuids) == 0:
        raise ValueError(f"No scene found. Is there any directory inside '{data_dir}'?")

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

    if(num_train < 0 or num_train > num_scenes or
       num_validation < 0 or num_validation > num_scenes or
       num_train < 0 or num_train > num_scenes):
        raise ValueError("Invalide data split proportion")

    print("Splitting method: random")

    random.shuffle(scenes_uuids)

    # list order: train     |   validation  | test
    # indices:    0  ...      num_train       num_train + num_validation
    
    test_start_id = num_train + num_validation

    write_lines(os.path.join(output_dir, "all_list.txt"),
                scenes_uuids)
    write_lines(os.path.join(output_dir, "training_list.txt"),
                scenes_uuids[:num_train])
    write_lines(os.path.join(output_dir, "validation_list.txt"),
                scenes_uuids[num_train:test_start_id])
    write_lines(os.path.join(output_dir, "test_list.txt"),
                scenes_uuids[test_start_id:])


def main():
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--data_root", type=str, help="Le répertoire qui contient un répertoire UUID par scène.", required=True)
    parser.add_argument("-o", "--output_dir", type=str, help="Le répertoire où créer les fichiers de split.", required=True)
    parser.add_argument("-v", "--validation_percent", type=float, default=0.05, help='Pourcentage de scènes à utiliser pour la validation (entre 0.0 et 1.0).')
    parser.add_argument("-t", "--test_percent", type=float, default=0.05, help='Pourcentage de scènes à utiliser pour les tests (entre 0.0 et 1.0).')
    parser.add_argument("-n", "--max_scenes", metavar="scene_count", type=int, default=-1, help="Nombre de scènes maximum à utiliser")
    args = parser.parse_args()

    generate_validation_split_files(args.data_root, args.output_dir, args.validation_percent, args.test_percent, args.max_scenes)


if __name__ == "__main__":
    main()
