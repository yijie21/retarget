# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    Virtual "mocap" via the MANO hand model (manopth).

    Instead of recording a real human hand (README Step 2), this samples MANO poses
    and exports fingertip trajectories in the SAME format GeoRT's trainer consumes:
    an [T, 21, 3] array of keypoints (meters) in the palm frame, with fingertips at
    MediaPipe indices 4/8/12/16/20 -- so the existing config (e.g. wuji.json) and
    trainer run unchanged.

    Why this works: the trainer only reads the 5 fingertips, turns each finger's
    points into a voxel-resampled point cloud (so coverage matters, not dwell time),
    and resamples each finger INDEPENDENTLY -- so the pinch loss fires wherever two
    fingers' workspaces overlap, not on recorded pinch frames. We therefore just need
    broad, correctly-framed fingertip coverage with overlapping thumb/finger regions.

    IMPORTANT: This script must run in an environment that has manopth + chumpy
    installed (chumpy is incompatible with numpy>=1.24, so use a dedicated env --
    see the repo notes). It deliberately does NOT import geort/sapien, so it can run
    in that isolated env. It writes straight to <repo>/data/<name>.npy.

    Usage (in the `mano` env):
        python geort/mocap/mano_mocap.py --name mano_right --n_samples 8000

    Then train (in the `geort` env):
        python geort/trainer.py -hand wuji -human_data mano_right -ckpt_tag wuji_mano
'''

import argparse
import os
from pathlib import Path

import numpy as np
import torch


# MANO right-hand fingertip VERTEX indices (used only if manopth returns the 16
# skeleton joints without tips). Order: thumb, index, middle, ring, pinky.
MANO_TIP_VIDS = [745, 317, 444, 556, 673]

# MANO's native 16-joint order is: 0 wrist, 1-3 index, 4-6 middle, 7-9 pinky,
# 10-12 ring, 13-15 thumb. We remap to the MediaPipe 21-joint layout below.
# MediaPipe: 0 wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
# (index 4/8/12/16/20 are the tips).
MANO16_TO_MP = {
    "wrist": 0,
    "thumb": [13, 14, 15],   # + tip
    "index": [1, 2, 3],      # + tip
    "middle": [4, 5, 6],     # + tip
    "ring": [10, 11, 12],    # + tip
    "pinky": [7, 8, 9],      # + tip
}
MP_TIP_IDX = [4, 8, 12, 16, 20]


def repo_root():
    return Path(__file__).resolve().parents[2]


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-9)


def build_mano_layer(mano_root, ncomps, side, flat_hand_mean):
    from manopth.manolayer import ManoLayer
    layer = ManoLayer(
        mano_root=str(mano_root),
        use_pca=True,
        ncomps=ncomps,
        flat_hand_mean=flat_hand_mean,
        side=side,
    )
    return layer


def sample_poses(n_samples, ncomps, pca_std, seed):
    '''Broad gaussian over the PCA pose space; global orientation fixed to zero so
    every sample shares one palm frame. Larger pca_std -> wider workspace coverage
    (and more thumb opposition -> more pinch overlap).'''
    g = torch.Generator().manual_seed(seed)
    global_orient = torch.zeros(n_samples, 3)
    pca = torch.randn(n_samples, ncomps, generator=g) * pca_std
    pose = torch.cat([global_orient, pca], dim=1)
    betas = torch.zeros(n_samples, 10)  # mean shape
    return pose, betas


def to_mediapipe_layout(verts, joints):
    '''Return [B, 21, 3] in MediaPipe order regardless of whether manopth gave 16 or
    21 joints. verts/joints are torch tensors [B, *, 3] in the model's native units.'''
    B = joints.shape[0]
    if joints.shape[1] == 21:
        # manopth already reorders to OpenPose/MediaPipe layout (tips at 4/8/12/16/20).
        return joints
    # Otherwise assemble from the 16 skeleton joints + fingertip vertices.
    tips = verts[:, MANO_TIP_VIDS, :]  # [B,5,3] thumb,index,middle,ring,pinky
    out = torch.zeros(B, 21, 3, dtype=joints.dtype)
    out[:, 0] = joints[:, MANO16_TO_MP["wrist"]]
    for fi, name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        base = 1 + fi * 4
        out[:, base:base + 3] = joints[:, MANO16_TO_MP[name]]
        out[:, base + 3] = tips[:, fi]
    return out


def autoscale_to_meters(kp):
    '''kp: [T,21,3]. MANO output may be meters or millimeters depending on the layer
    version; detect from middle-finger length and convert to meters.'''
    middle_len = np.linalg.norm(kp[:, 12] - kp[:, 0], axis=1).mean()  # wrist->middle tip
    if middle_len > 1.0:  # >1m is impossible for a hand -> it's millimeters
        kp = kp / 1000.0
        middle_len /= 1000.0
    return kp, middle_len


def to_palm_frame(kp):
    '''Rotate keypoints into GeoRT's convention using the MEAN pose geometry:
        +Z: palm center -> middle MCP, +Y: palm center -> thumb, +X: palm normal.
    Origin = palm center (mean of wrist + finger MCPs). Same transform for all frames
    (global orientation was fixed), so it is a rigid relabel of axes.'''
    J = kp.mean(axis=0)  # [21,3]
    mcp_ids = [5, 9, 13, 17]  # index/middle/ring/pinky MCP
    palm_center = kp[:, [0] + mcp_ids].mean(axis=(0, 1))  # wrist + MCPs, over all frames
    z = normalize(J[9] - palm_center)        # palm -> middle
    y_raw = normalize(J[2] - palm_center)    # toward thumb MCP
    x = normalize(np.cross(y_raw, z))        # palm normal
    y = normalize(np.cross(z, x))            # re-orthogonalized palm -> thumb
    R = np.stack([x, y, z], axis=1)          # columns are the new axes
    local = (kp - palm_center) @ R           # [T,21,3]
    return local.astype(np.float32), (palm_center, R)


def report_stats(kp, ref_path=None):
    flat = kp.reshape(-1, 3)
    print("  workspace extent (m):", (flat.max(0) - flat.min(0)).round(3),
          "| min", flat.min(0).round(3), "max", flat.max(0).round(3))
    # tip-from-wrist sanity: tip must be the farthest joint of each finger
    for fi, (name, tip) in enumerate(zip(["thumb", "index", "middle", "ring", "pinky"], MP_TIP_IDX)):
        finger = list(range(1 + fi * 4, 1 + fi * 4 + 4))
        d = np.linalg.norm(kp[:, finger] - kp[:, [0]], axis=2).mean(0)
        ok = "OK" if d.argmax() == 3 else "WARN tip not farthest!"
        print(f"  {name}: mean joint-from-wrist dist {d.round(3)} [{ok}]")
    # pinch overlap (what the pinch loss can use)
    for a, b, lbl in [(4, 8, "thumb-index"), (4, 12, "thumb-middle"), (4, 16, "thumb-ring"), (4, 20, "thumb-pinky")]:
        dist = np.linalg.norm(kp[:, a] - kp[:, b], axis=1)
        print(f"  {lbl}: min {dist.min():.3f}m, frames<1.5cm {int((dist < 0.015).sum())}/{len(dist)}")
    if ref_path and os.path.exists(ref_path):
        ref = np.load(ref_path).reshape(-1, 3)
        print("  [ref human_alex extent (m)]:", (ref.max(0) - ref.min(0)).round(3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default="mano_right", help="output dataset name -> data/<name>.npy")
    p.add_argument("--n_samples", type=int, default=8000)
    p.add_argument("--ncomps", type=int, default=45, help="MANO PCA components (<=45)")
    p.add_argument("--pca_std", type=float, default=2.0, help="std of PCA sampling; larger = wider coverage")
    p.add_argument("--side", type=str, default="right", choices=["right", "left"])
    p.add_argument("--flat_hand_mean", action="store_true", help="use flat mean hand instead of MANO's curled mean")
    p.add_argument("--mano_root", type=str, default=None, help="dir with MANO_RIGHT.pkl (default: assets/mano_v1_2/models)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    mano_root = Path(args.mano_root) if args.mano_root else (repo_root() / "assets" / "mano_v1_2" / "models")
    print(f"MANO root: {mano_root}")
    layer = build_mano_layer(mano_root, args.ncomps, args.side, args.flat_hand_mean)

    pose, betas = sample_poses(args.n_samples, args.ncomps, args.pca_std, args.seed)
    print(f"Sampling {args.n_samples} MANO poses (ncomps={args.ncomps}, pca_std={args.pca_std})...")
    with torch.no_grad():
        verts, joints = layer(pose, betas)

    kp = to_mediapipe_layout(verts, joints).cpu().numpy().astype(np.float64)
    kp, middle_len = autoscale_to_meters(kp)
    print(f"Detected middle-finger length ~ {middle_len:.3f} m")

    kp, _ = to_palm_frame(kp)

    print("Sampled-data stats:")
    report_stats(kp, ref_path=str(repo_root() / "data" / "human_alex.npy"))

    out_dir = repo_root() / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.name}.npy"
    np.save(out_path, kp)
    print(f"\nSaved {kp.shape} -> {out_path}")
    print(f"Train with:\n  python geort/trainer.py -hand wuji -human_data {args.name} -ckpt_tag wuji_mano")


if __name__ == "__main__":
    main()
