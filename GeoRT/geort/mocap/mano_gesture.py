# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    Generate a recognizable MANO gesture sequence for the side-by-side comparison demo.

    Unlike mano_mocap.py (random poses for training coverage), this produces a smooth,
    legible motion -- curl each finger in turn, then make a fist -- so you can visually
    check that the retargeted robot hand mimics the same gesture. It saves BOTH:
      data/<name>.npy        : keypoints [T,21,3], palm frame, meters (for retargeting)
      data/<name>_mano.npz   : verts [T,778,3] + faces [F,3] (the MANO mesh, same frame)

    Run in the `mano` env:
        python geort/mocap/mano_gesture.py --name mano_gesture

    Then compare in the `geort` env:
        python geort/mocap/compare_viser.py -hand wuji -ckpt_tag wuji_mano -data mano_gesture
'''

import argparse
import os
import sys

import numpy as np
import torch

# Import the standalone helpers directly by path: the `mano` env has manopth but not the
# installed `geort` package, so we must avoid triggering geort/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mano_mocap import to_mediapipe_layout, repo_root, normalize

# MANO non-PCA pose layout: pose[0:3]=global, then 15 joints x 3 axis-angle, ordered
# index(3 joints), middle, pinky, ring, thumb. Joint k -> pose[3+3k : 3+3k+3].
FINGER_MANO_JOINTS = {
    "thumb": [12, 13, 14],
    "index": [0, 1, 2],
    "middle": [3, 4, 5],
    "ring": [9, 10, 11],
    "pinky": [6, 7, 8],
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
MP_TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
MAX_CURL = 1.5  # radians applied to each finger joint at full curl


def make_pose(curl_amounts, curl_axes):
    '''curl_amounts: dict finger->[0,1]. Returns a [48] MANO pose (non-PCA), global=0.'''
    pose = np.zeros(48, dtype=np.float32)
    for finger in FINGER_ORDER:
        axis, sign = curl_axes[finger]
        ang = curl_amounts[finger] * MAX_CURL * sign
        for j in FINGER_MANO_JOINTS[finger]:
            pose[3 + 3 * j + axis] = ang
    return pose


def find_curl_axes(layer, betas):
    '''Empirically pick, per finger, the (axis, sign) that curls it most (brings the
    fingertip closest to the wrist). Robust to MANO's per-joint axis conventions.'''
    axes = {}
    for finger in FINGER_ORDER:
        best = None
        for axis in range(3):
            for sign in (1.0, -1.0):
                amounts = {f: 0.0 for f in FINGER_ORDER}
                amounts[finger] = 1.0
                pose = make_pose(amounts, {finger: (axis, sign), **{f: (0, 1.0) for f in FINGER_ORDER if f != finger}})
                with torch.no_grad():
                    v, j = layer(torch.from_numpy(pose).unsqueeze(0), betas)
                kp = to_mediapipe_layout(v, j)[0].numpy()
                d = np.linalg.norm(kp[MP_TIP[finger]] - kp[0])  # tip -> wrist
                if best is None or d < best[0]:
                    best = (d, axis, sign)
        axes[finger] = (best[1], best[2])
    return axes


def ease(a, b, n):
    '''Cosine-eased interpolation from a to b over n frames (a..b inclusive-ish).'''
    t = (1 - np.cos(np.linspace(0, np.pi, n))) / 2  # 0..1 smooth
    return [(1 - x) * np.array(a) + x * np.array(b) for x in t]


def build_keyframes():
    '''Per-finger curl amounts [thumb,index,middle,ring,pinky] at each held pose.'''
    O = [0, 0, 0, 0, 0]
    seq = [O]
    for i in range(5):                 # curl each finger in turn, returning to open
        k = O.copy(); k[i] = 1
        seq += [k, O]
    seq += [[1, 1, 1, 1, 1], O]        # fist, then open
    return seq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default="mano_gesture")
    p.add_argument("--seg_frames", type=int, default=18, help="frames per interpolated segment")
    p.add_argument("--mano_root", type=str, default=None)
    args = p.parse_args()

    mano_root = args.mano_root or str(repo_root() / "assets" / "mano_v1_2" / "models")
    from manopth.manolayer import ManoLayer
    layer = ManoLayer(mano_root=mano_root, use_pca=False, flat_hand_mean=True, side="right")
    betas = torch.zeros(1, 10)
    faces = layer.th_faces.cpu().numpy().astype(np.int32)

    print("Detecting per-finger curl axes...")
    curl_axes = find_curl_axes(layer, betas)
    print("  ", {f: curl_axes[f] for f in FINGER_ORDER})

    # Build the full per-frame curl trajectory.
    keys = build_keyframes()
    traj = []
    for a, b in zip(keys[:-1], keys[1:]):
        traj += ease(a, b, args.seg_frames)
    print(f"Generating {len(traj)} frames ...")

    poses = []
    for amounts_vec in traj:
        amounts = {f: float(amounts_vec[i]) for i, f in enumerate(FINGER_ORDER)}
        poses.append(make_pose(amounts, curl_axes))
    poses = torch.from_numpy(np.stack(poses))

    with torch.no_grad():
        verts, joints = layer(poses, betas.expand(len(poses), -1))
    kp = to_mediapipe_layout(verts, joints).cpu().numpy().astype(np.float64)   # [T,21,3]
    verts = verts.cpu().numpy().astype(np.float64)                            # [T,778,3]

    # Scale to meters (manopth returns mm here) using the same factor for kp and verts.
    if np.linalg.norm(kp[:, 12] - kp[:, 0], axis=1).mean() > 1.0:
        kp /= 1000.0
        verts /= 1000.0

    # Palm-frame transform from the first (flat) frame; apply to BOTH kp and verts so the
    # two meshes share the convention used elsewhere (+Y thumb, +Z middle, +X normal).
    J = kp[0]
    palm_center = kp[0, [0, 5, 9, 13, 17]].mean(axis=0)
    z = normalize(J[9] - palm_center)
    yb = normalize(J[2] - palm_center)
    x = normalize(np.cross(yb, z))
    y = normalize(np.cross(z, x))
    R = np.stack([x, y, z], axis=1)
    kp = ((kp - palm_center) @ R).astype(np.float32)
    verts = ((verts - palm_center) @ R).astype(np.float32)

    out_dir = repo_root() / "data"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / f"{args.name}.npy", kp)
    np.savez(out_dir / f"{args.name}_mano.npz", verts=verts, faces=faces)
    print(f"Saved keypoints {kp.shape} -> data/{args.name}.npy")
    print(f"Saved MANO mesh verts {verts.shape}, faces {faces.shape} -> data/{args.name}_mano.npz")
    print(f"\nCompare with:\n  python geort/mocap/compare_viser.py -hand wuji -ckpt_tag wuji_mano -data {args.name}")


if __name__ == "__main__":
    main()
