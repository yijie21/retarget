# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    Convert a FürElise piece (rcwang/for_elise) into GeoRT's comparison-demo format.

    The dataset's per-piece motion.pkl already contains, per hand, the MANO joints
    (T,21,3, MediaPipe layout) AND verts (T,778,3), so NO manopth is needed -- this runs
    in the `geort` env directly. We re-express each frame in its palm frame (removing the
    global hand motion across the keyboard) and save keypoints + mesh for compare_viser.

    Download a piece first (no full 47GB download needed):
        python geort/mocap/furelise_fetch.py --piece 65 --out <dir>

    Then convert and compare:
        python geort/mocap/furelise_to_geort.py --pkl <dir>/motion.pkl --hand right --name furelise_65
        python geort/mocap/compare_viser.py -hand wuji -ckpt_tag wuji_mano -data furelise_65
'''

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mano_common import autoscale_to_meters, palm_frame_per_frame, report, save_outputs

# Default faces shipped with the dataset's visualizer (downloaded next to the pieces).
DEFAULT_FACES = "/mnt/yijie/storage/hf_datasets/geort/furelise/mano_faces.pkl"


def repo_data_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "data"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", type=str, required=True, help="path to a FürElise piece motion.pkl")
    p.add_argument("--hand", type=str, default="right", choices=["right", "left"])
    p.add_argument("--name", type=str, default="furelise")
    p.add_argument("--faces", type=str, default=DEFAULT_FACES, help="mano_faces.pkl (dict with *_faces)")
    p.add_argument("--start", type=int, default=0, help="start frame")
    p.add_argument("--end", type=int, default=-1, help="end frame (-1 = all)")
    p.add_argument("--stride", type=int, default=2, help="frame stride (orig ~60fps; 2 -> ~30fps)")
    args = p.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    hand = data[args.hand]
    joints = np.asarray(hand["joints"], dtype=np.float64)              # [T,21,3]
    verts = np.asarray(hand["mano_params"]["verts"], dtype=np.float64)  # [T,778,3]

    end = len(joints) if args.end < 0 else args.end
    sl = slice(args.start, end, args.stride)
    joints, verts = joints[sl], verts[sl]
    print(f"Loaded {args.hand} hand: {joints.shape[0]} frames after slicing [{args.start}:{end}:{args.stride}]")

    # Faces: prefer dataset faces; the verts use the standard MANO 778-vertex order.
    with open(args.faces, "rb") as f:
        fdict = pickle.load(f)
    faces = fdict[f"{args.hand}_faces"]

    joints, verts, mlen = autoscale_to_meters(joints, verts)
    print(f"  middle-finger length ~ {mlen:.3f} m")
    kp, verts = palm_frame_per_frame(joints, verts)
    report(kp)

    save_outputs(repo_data_root(), args.name, kp, verts, faces)


if __name__ == "__main__":
    main()
