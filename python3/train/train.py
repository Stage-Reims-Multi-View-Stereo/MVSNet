#!/usr/bin/env python3

import subprocess as subp
import argparse 
import os
import json

# Pour argparse, s"assurer qu"un argument est bien un répertoire
def dir(string):
    if os.path.isdir(string):
        return string
    else:
        raise ValueError(string)

DESC = """
Entraîner MVSNet avec les paramètres stockés dans un dossier.
Permet de sauvegarder les paramètres et savoir quel dossier correspond à quels paramètres.
Les paramètres sont stockés dans un fichier JSON, regarder les exemples pour les comprendre.
La clé [dataset][type] peut valoir soit "blendedmvs", soit "dtu", soit "eth3d", soit "revery".
Si [dataset][type] vaut "revery", alors le répertoire du modèle doit aussi contenir en plus
le répertoire "cams" qui stocke les paramètres de caméra, et aussi les fichiers "testing_list.txt",
"training_list.txt", et "validation_list.txt".

Cela créé les répertoires "tf_model", "tf_log" dans le répertoire du modèle.
"""

def main():
    parser = argparse.ArgumentParser(description=DESC, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-m", "--model_dir", type=dir,
                        help="Directory containing the file 'training_settings.json'",
                        required=True)
    parser.add_argument("-s", "--sif", type=str,
                        help="Fichier .sif qui contient l'image vers le conteneur Singularity de MVSNet",
                        required=True)
    args = parser.parse_args()
    
    if not os.path.isfile(args.sif): raise ValueError(f"File not found: {args.sif}")
    
    settings_json = json.load(open(os.path.join(args.model_dir, "training_settings.json")))
    num_gpus = int(settings_json["environment"]["num_gpus"])
    
    # Faire attention à utiliser le chemin du script,
    # Le script n'est pas forcément appelé depuis le répertoire qui le contient
    current_script_dir = os.path.dirname(__file__)
    mvsnet_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
    
    
    slurm_stdout_path = os.path.join(args.model_dir, "slurm.stdout.txt")
    
    # clear temp output file first
    os.remove(slurm_stdout_path)
    
    sbatch_options = [
        f"--partition=long",
        f"--gres=gpu:{num_gpus}",
        f"--output", slurm_stdout_path
    ]
    
    # Binds:
    #   /dataset: Dataset root (read-only)
    #   /mvsnet: Source code MVSNet qui contient train.py, test.py, etc (read-only)
    #   /model: Model root (read/write)
    # Paths dans le JSON relatifs au dossier qui contient le JSON
    singularity_binds = [
        f"{os.path.abspath(os.path.relpath(settings_json['dataset']['path'], args.model_dir))}:/dataset:ro",
        f"{mvsnet_dir}:/mvsnet:ro",
        f"{os.path.abspath(args.model_dir)}:/model:rw",
    ]
    
    singularity_options = [
        "--bind", ",".join(singularity_binds),
        "--pwd", "/mvsnet/mvsnet",
        "--no-home"
    ]
    
    # python2 train.py --train_revery --revery_data_root /data/revery/ --regularization '3DCNNs' 
    # --max_w 1024 --max_h 576 --max_d 128 --model_folder /data/tf_model_revery_romeo_D
    # --num_gpus 4 --online_augmentation --refinement --log_folder /data/tf_model_revery_romeo_D/tf_log
    
    # str(int(X)) est juste pour s'assurer que la valeur est bien entière, et la convertir en string
    # car subprocess.check_call() attend une liste de strings uniquement.
    
    # Les chemins sont dans le conteneur c'est pour ça qu'on utilise directement /model et pas le chemin donné en argument,
    # qui sera monté ici
    
    mvsnet_command = [
        "python2", "train.py",
        "--max_w", str(int(settings_json["volume"]["max_w"])),
        "--max_h", str(int(settings_json["volume"]["max_h"])),
        "--max_d", str(int(settings_json["volume"]["max_d"])),
        "--regularization", settings_json["regularization"],
        "--view_num", str(int(settings_json["num_views"])),
        "--log_folder", "/model/tf_log",
        "--model_folder", "/model/tf_model",
        "--num_gpus", str(num_gpus),
        "--ckpt_step", str(int(settings_json["hyperparameters"]["steps"])),
        "--batch_size", str(int(settings_json["hyperparameters"]["batch_size"])),
    ]
    
    dataset_type = settings_json["dataset"]["type"]
    if dataset_type == "revery":
        mvsnet_command += [
            "--train_revery",
z            "--revery_cams_dir", "/model/cams"
        ]
    elif dataset_type == "blendedmvs":
        mvsnet_command += [
            "--train_blendedmvs",
            "--blendedmvs_data_root", "/dataset"
        ]
    else:
        raise ValueError(f"Invalid dataset type: '{dataset_type}'")
    
    if settings_json["flags"]["refinement"]: mvsnet_command += ["--refinement"]
    if settings_json["flags"]["online_augmentation"]: mvsnet_command += ["--online_augmentation"]
    
    # Le job prend en option littéralement la commande à lancer ainsi que ses arguments
    # On rajoute 'time' pour mesurer le temps
    # Mais pas dans le conteneur (time n'y est pas installé, fix?)
    job_options = ["time", "singularity", "exec"] + singularity_options + [args.sif] + mvsnet_command
    
    job_script_path = os.path.join(current_script_dir, "train.slurm.sh")
    subp_command = ["sbatch"] + sbatch_options + [job_script_path] + job_options
    
    print("---")
    print("Running command:")
    print(" ".join(subp_command))
    print("---")
    
    subp.check_call(subp_command)

    print("Replacing Python process by 'tail -F':")
    # execlp: Pour pouvoir utiliser la commande sans chemin absolu
    # Permet de suivre la sortie du processus
    # Remplace le processus Python
    # Peut être ctrl^c sans annuler le job
    os.execlp("tail", "tail", "-F", slurm_stdout_path)
    
if __name__ == "__main__":
    main()
