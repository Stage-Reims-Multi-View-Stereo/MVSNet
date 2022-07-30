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
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description="Train MVSNet with custom parameters.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset_root",    type=dir,   default=None,  help="Directory containing training_list.txt, UUIDS scenes directories, cams/pair.txt and cams/000000[0-16]_cam.txt.", required=True)
parser.add_argument("-o", "--model_dir",       type=str,   default=None,  help="Output network model directory.", required=True)
parser.add_argument("-c", "--cams_dir",        type=dir,   default=None,  help="Directory containing *_cam.txt files, and pair.txt file.", required=True)
parser.add_argument("-W", "--image_width",     type=int,   default=None,  help="Resize every image width when loaded.")
parser.add_argument("-H", "--image_height",    type=int,   default=None,  help="Resize every image height when loaded.")
parser.add_argument("-r", "--r_mvsnet",        action="store_true",       help="Use R-MVSNet instead of MVSNet.")
parser.add_argument("-k", "--ckpt_step",       type=int,   default=15000, help="Same as MVSNet train.py parameter.")
parser.add_argument("-s", "--sample_scale",    type=float, default=0.25,  help="Same as MVSNet train.py parameter.")
parser.add_argument("-n", "--num_gpus",        type=int,   default=1,     help="Same as MVSNet train.py parameter.")
parser.add_argument("-v", "--view_num",        type=int,   default=5,     help="Same as MVSNet train.py parameter.")
parser.add_argument("-x", "--max_w",           type=int,   default=640,   help="Same as MVSNet train.py parameter.")
parser.add_argument("-y", "--max_h",           type=int,   default=512,   help="Same as MVSNet train.py parameter.")
parser.add_argument("-z", "--max_d",           type=int,   default=128,   help="Same as MVSNet train.py parameter.")
parser.add_argument("-b", "--batch_size",      type=int,   default=1,     help="Same as MVSNet train.py parameter.")
args = parser.parse_args()

if args.r_mvsnet:
    regularization = "3DCNNs"
else:
    regularization = "GRU"

# On passe certaines options en variable d"environnement
# Ce qui permet de modifier moins de code de MVSNet et de devoir modifier les arguments
if args.image_width is not None:
    os.environ["REVERY_IMAGE_WIDTH"] = str(args.image_width)
if args.image_height is not None:
    os.environ["REVERY_IMAGE_HEIGHT"] = str(args.image_height)

cam_settings = json.load(open(os.path.join(args.cams_dir, "generation_settings.json")))

os.environ["REVERY_WRITE_PNG_DEPTH"] = str(1)
os.environ["REVERY_READ_PNG_DEPTH"] = str(1)
os.environ["REVERY_DISPARITY_NUMERATOR"] = str(cam_settings["disparity_numerator"])

train_args = [
    "--revery_data_root", args.dataset_root,
    "--revery_cams_dir", str(args.cams_dir),
    "--train_revery",
    "--log_folder", os.path.join(args.model_dir, "train_log"),
    "--model_folder", args.model_dir,
    "--ckpt_step", str(args.ckpt_step),
    "--view_num", str(args.view_num),
    "--max_d", str(args.max_d),
    "--max_w", str(args.max_w),
    "--max_h", str(args.max_h),
    "--sample_scale", str(args.sample_scale),
    "--regularization", regularization,
    "--num_gpus", str(args.num_gpus),
    "--batch_size", str(args.batch_size)
]

subp.check_call(["/usr/bin/env", "python2.7", "../mvsnet/train.py"] + train_args)
