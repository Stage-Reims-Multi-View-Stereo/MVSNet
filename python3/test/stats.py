import pandas as pd
import cv2
from typing import Callable
import numpy as np

def imread(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Image '{path}' could not be loaded.")
    return image

def compute_dataset_stats(dataset_paths: list, gen_mesures: Callable[[np.ndarray, np.ndarray], dict]):
    """
    Charge les données statistique du dataset.
    
    Args:
        dataset_path (list):
            La liste des scènes, chaque scène étant une liste de paires de chemins vers les depths maps.
        stats_fun (function):
            Fonction prenant en arguments les deux images, et retourne un dictionnaire contenant les mesures souhaitées.
            Doit toujours retourner les mêmes clés, sinon erreur.
    
    Returns:
        Un DataFrame ayant comme colonnes 'scene', 'cam', plus une colonne par mesure.
    """
    
    df_data = {
        "scene": [],
        "cam": []
    }
    
    for scene_id, scene_paths in enumerate(dataset_paths):
        
        for cam_id, (truth_path, inferred_path) in enumerate(scene_paths):
                
            truth_image = imread(truth_path)
            inferred_image = imread(inferred_path)
            
            # Most of the time, inferred depth is samaller than original depth image
            truth_image = cv2.resize(truth_image, dsize=inferred_image.shape[::-1], interpolation=cv2.INTER_LINEAR)
            
            mesures = gen_mesures(truth_image, inferred_image)
            
            for mkey, mvalue in mesures.items():
                
                # Only add missing key if it is the first image of the dataset
                if (not mkey in df_data) and (not df_data["cam"]):
                    raise ValueError(f"Key does not appear in every mesure: '{mkey}'")
                
                df_data[mkey].insert(mvalue)
       
            df_data["scene"].insert(scene_id)
            df_data["cam"].insert(cam_id)
    
    df = pd.DataFrame(df_data)
    return df