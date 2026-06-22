# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    Convert a GRAB sequence (otaheri/GRAB) into GeoRT's comparison-demo format.

    GRAB stores per-frame MANO *parameters* (not verts), so this runs manopth in the
    `mano` env. Each per-sequence .npz, when loaded GRAB-style, is a dict with keys like
    'n_comps', 'framerate', 'rhand', 'lhand', 'object', ... and rhand['params'] holds
    'global_orient' [N,3], 'hand_pose' [N,K] (K<45 => PCA coeffs; K==45 => axis-angle),
    and 'transl'. GRAB's MANO uses flat_hand_mean=True; per-subject shape comes from a
    'vtemp' template -- we approximate it with the mean shape (betas=0), which is fine for
    a finger-articulation comparison.

    GRAB requires a (free) licensed download from https://grab.is.tue.mpg.de. This adapter
    has been validated against real GRAB sequences (subject s2): it reads the per-frame
    'fullpose' (45-d axis-angle) when present, else 'hand_pose' (PCA). It prints the
    detected structure on load; if a field name/shape differs in your copy, adjust the
    small parsing block in load_grab_seq().

    Usage (in the `mano` env):
        python geort/mocap/grab_to_geort.py --npz <grab>/s1/<seq>.npz --hand right --name grab_demo
    Then (in the `geort` env):
        python geort/mocap/compare_viser.py -hand wuji -ckpt_tag wuji_mano -data grab_demo
'''

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mano_mocap import to_mediapipe_layout, repo_root
from _mano_common import autoscale_to_meters, palm_frame_per_frame, report, save_outputs


def load_grab_seq(npz_path, hand):
    '''Return (global_orient [N,3], hand_pose [N,K], n_comps) for the requested hand.'''
    raw = np.load(npz_path, allow_pickle=True)
    data = {k: raw[k].item() if raw[k].dtype == object else raw[k] for k in raw.files}
    n_comps = int(np.asarray(data.get("n_comps", 24)).item()) if "n_comps" in data else None

    key = "rhand" if hand == "right" else "lhand"
    if key not in data:
        raise SystemExit(f"'{key}' not found. Top-level keys: {list(data.keys())}")
    params = data[key]["params"] if isinstance(data[key], dict) and "params" in data[key] else data[key]
    print(f"  {key}.params keys: {list(params.keys())}")
    go = np.asarray(params["global_orient"], dtype=np.float64)
    # Prefer 'fullpose' (full 45-d axis-angle finger pose) -> no PCA-basis assumptions.
    # Fall back to 'hand_pose' (PCA coeffs) if fullpose is absent.
    if "fullpose" in params:
        hp = np.asarray(params["fullpose"], dtype=np.float64)[:, -45:]
    elif "hand_pose" in params:
        hp = np.asarray(params["hand_pose"], dtype=np.float64)
    else:
        raise SystemExit(f"No 'fullpose'/'hand_pose' in params: {list(params.keys())}")
    print(f"  global_orient {go.shape}, hand_pose {hp.shape}, n_comps={n_comps}")
    return go, hp, n_comps


def run_mano(go, hp, side, batch=512):
    from manopth.manolayer import ManoLayer
    K = hp.shape[1]
    use_pca = K < 45
    ncomps = K if use_pca else 45
    print(f"  manopth: use_pca={use_pca}, ncomps={ncomps}, flat_hand_mean=True, side={side}")
    layer = ManoLayer(mano_root=str(repo_root() / "assets" / "mano_v1_2" / "models"),
                      use_pca=use_pca, ncomps=ncomps, flat_hand_mean=True, side=side)
    faces = layer.th_faces.cpu().numpy().astype(np.int32)
    pose = np.concatenate([go, hp], axis=1).astype(np.float32)   # [N, 3+K]
    betas = torch.zeros(1, 10)
    all_kp, all_v = [], []
    for s in range(0, len(pose), batch):
        pe = torch.from_numpy(pose[s:s + batch])
        with torch.no_grad():
            v, j = layer(pe, betas.expand(len(pe), -1))
        all_kp.append(to_mediapipe_layout(v, j).cpu().numpy())
        all_v.append(v.cpu().numpy())
    return np.concatenate(all_kp).astype(np.float64), np.concatenate(all_v).astype(np.float64), faces


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True, help="path to a GRAB sequence .npz")
    p.add_argument("--hand", type=str, default="right", choices=["right", "left"])
    p.add_argument("--name", type=str, default="grab_demo")
    p.add_argument("--stride", type=int, default=4, help="GRAB is ~120fps; 4 -> ~30fps")
    args = p.parse_args()

    go, hp, _ = load_grab_seq(args.npz, args.hand)
    go, hp = go[::args.stride], hp[::args.stride]
    print(f"Using {len(go)} frames (stride {args.stride}).")

    kp, verts, faces = run_mano(go, hp, args.hand)
    kp, verts, mlen = autoscale_to_meters(kp, verts)
    print(f"  middle-finger length ~ {mlen:.3f} m")
    kp, verts = palm_frame_per_frame(kp, verts)
    report(kp)

    save_outputs(str(repo_root() / "data"), args.name, kp, verts, faces)


if __name__ == "__main__":
    main()
