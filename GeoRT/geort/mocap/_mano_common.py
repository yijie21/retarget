# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    Shared helpers for converting external MANO-pose sequence datasets into GeoRT's
    comparison-demo format. Pure NumPy so it works in both the `geort` env (furelise
    adapter, which has precomputed verts) and the `mano` env (grab adapter, which runs
    manopth). Output: data/<name>.npy [T,21,3] + data/<name>_mano.npz (verts, faces).
'''

import numpy as np

MP_TIP_IDX = [4, 8, 12, 16, 20]
MP_MCP_IDX = [5, 9, 13, 17]  # index/middle/ring/pinky MCP (MediaPipe layout)


def _norm(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-9)


def autoscale_to_meters(kp, verts):
    '''Detect mm vs m from middle-finger length and convert both arrays to meters.'''
    middle_len = np.linalg.norm(kp[:, 12] - kp[:, 0], axis=1).mean()
    if middle_len > 1.0:  # >1 m impossible for a hand -> millimeters
        return kp / 1000.0, (verts / 1000.0 if verts is not None else None), middle_len / 1000.0
    return kp, verts, middle_len


def palm_frame_per_frame(kp, verts):
    '''Re-express every frame in its own palm frame (+Y thumb, +Z middle, +X normal),
    removing global wrist translation/rotation so the hand stays put and only the finger
    articulation remains -- the right input for GeoRT retargeting and for a clean
    side-by-side. Applies the SAME per-frame transform to kp and verts.'''
    pc = kp[:, [0] + MP_MCP_IDX, :].mean(axis=1)        # [T,3] palm center
    z = _norm(kp[:, 9] - pc)                            # palm -> middle MCP
    yb = _norm(kp[:, 2] - pc)                           # toward thumb MCP
    x = _norm(np.cross(yb, z))                          # palm normal
    y = _norm(np.cross(z, x))                           # re-orthogonalized thumb dir
    R = np.stack([x, y, z], axis=-1)                    # [T,3,3], columns = axes
    kp_l = np.einsum('tij,tjk->tik', kp - pc[:, None, :], R)
    v_l = None
    if verts is not None:
        v_l = np.einsum('tij,tjk->tik', verts - pc[:, None, :], R)
        v_l = v_l.astype(np.float32)
    return kp_l.astype(np.float32), v_l


def report(kp):
    flat = kp.reshape(-1, 3)
    print("  frames:", len(kp), "| extent (m):", (flat.max(0) - flat.min(0)).round(3))
    for fi, (name, tip) in enumerate(zip(["thumb", "index", "middle", "ring", "pinky"], MP_TIP_IDX)):
        finger = list(range(1 + fi * 4, 1 + fi * 4 + 4))
        d = np.linalg.norm(kp[:, finger] - kp[:, [0]], axis=2).mean(0)
        if d.argmax() != 3:
            print(f"  WARNING: {name} tip is not the farthest joint -- check joint layout.")


def save_outputs(data_root, name, kp, verts, faces):
    import os
    os.makedirs(data_root, exist_ok=True)
    np.save(os.path.join(data_root, f"{name}.npy"), kp.astype(np.float32))
    np.savez(os.path.join(data_root, f"{name}_mano.npz"),
             verts=verts.astype(np.float32), faces=np.asarray(faces).astype(np.int32))
    print(f"Saved keypoints {kp.shape} -> data/{name}.npy")
    print(f"Saved MANO mesh verts {verts.shape}, faces {np.asarray(faces).shape} -> data/{name}_mano.npz")
    print(f"\nCompare with:\n  python geort/mocap/compare_viser.py -hand wuji -ckpt_tag wuji_mano -data {name}")
