import os
import glob
import sys


def print_error(msg):
    sys.stderr.write(msg + "\n")


def remove_extension(path):
    return os.path.splitext(path)[0]


def load_scene_paths(scene_dir: str):
    """
    Extrait les paires (truth, inferred) qui contient les chemins vers les images
    de la vérité terrain et inférées.
    
    Args:
        scene_dir (str): Le chemin de la scène qui doit contenir le dossier "depths_mvsnet/" et "rendered_depth_maps/".
    
    Returns:
        Une liste sous la forme suivante :
            [["truth1.pfm", "inferred1.pfm"],
             ["truth2.pfm", "inferred2.pfm"],
             ...]
    """
    
    ground_truth_pathlist = glob.glob(os.path.join(scene_dir, "rendered_depth_maps", "*.pfm"))
    ground_truth_pathlist.sort()
    scene = []
    
    if not ground_truth_pathlist:
        raise ValueError(f"No PFM file found in scene path '{scene_dir}'")
        
    for ground_truth_path in ground_truth_pathlist:
        cam_id = remove_extension(os.path.basename(ground_truth_path))
        
        if not cam_id.isdigit():
            raise ValueError(f"Found PFM file which the name is not a camera id: '{ground_truth_path}'")
    
        inferred_path = os.path.join(scene_dir, "depths_mvsnet", cam_id + "_init.pfm")

        if not os.path.isfile(inferred_path):
            raise ValueError(f"Inferred depth path '{inferred_path}' does not exist")

        scene.append([ground_truth_path, inferred_path])
    
    return scene


def load_dataset_paths(dataset_root: str, txt = "testing_list.txt"):
    """
    Arguments:
        dataset_root (str): Le chemin répertoire qui contient 'testing_list.txt'.
        txt (str): Le nom du fichier qui contient la liste des scènes, un nom de scène correspondant à un nom de dossier par ligne.
        
    Returns:
        La liste des paires (chemin truth, chemin inferred) pour chaque caméra de chaque scène.
    """
    
    txt_path = os.path.join(dataset_root, txt)
    
    if not os.path.isfile(txt_path):
        raise ValueError(f"File '{txt}' not found in '{dataset_root}'.")
    
    lines = tuple(open(txt_path, "r"))
    
    dataset_paths = []
    
    for line in lines:
        scene_name = line.strip()
        dataset_paths.append(load_scene_paths(os.path.join(dataset_root, scene_name)))        
    
    if not dataset_paths:
        raise ValueError(f"Training file '{txt}' is empty.")

    return dataset_paths
