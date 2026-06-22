# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    Frame-alignment diagnostic (run in the `geort` env).

    GeoRT's Chamfer loss compares the human fingertip cloud (palm frame) DIRECTLY
    against the robot fingertip cloud computed in the URDF's base_link frame. For
    training to make sense those two clouds must live in the SAME convention:
        +X palm normal, +Y palm->thumb, +Z palm->middle, origin ~ palm center.

    This script samples random robot qpos, computes the fingertip cloud in base_link
    frame, and prints its per-axis extent next to the MANO human cloud's. If the
    "long" axis (fingers) or the thumb direction disagree, the wuji base_link is not
    in the convention and you should add a virtual base link (rotation) to the URDF
    and point config "base_link" at it.

    Usage:
        python geort/mocap/check_alignment.py --hand wuji --human mano_right
'''

import argparse

import numpy as np

from geort.utils.config_utils import get_config
from geort.utils.path import get_human_data
from geort.env.hand import HandKinematicModel


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
MP_TIP_IDX = [4, 8, 12, 16, 20]


def robot_tip_cloud(config, n=20000):
    model = HandKinematicModel.build_from_config(config, render=False)
    info_links = [f["link"] for f in config["fingertip_link"]]
    info_off = [f["center_offset"] for f in config["fingertip_link"]]
    info_names = [f["name"] for f in config["fingertip_link"]]
    model.initialize_keypoint(keypoint_link_names=info_links, keypoint_offsets=info_off)

    lo, hi = model.get_joint_limit()
    lo, hi = np.array(lo), np.array(hi)
    pts = {name: [] for name in info_names}
    for _ in range(n):
        qpos = np.random.uniform(0, 1, len(lo)) * (hi - lo) + lo
        vec = model.keypoint_from_qpos(qpos, ret_vec=True)  # [n_finger,3], config order
        for i, name in enumerate(info_names):
            pts[name].append(vec[i])
    return {k: np.array(v) for k, v in pts.items()}, info_names


def axis_report(tag, flat):
    ext = flat.max(0) - flat.min(0)
    print(f"  [{tag}] extent (m) X={ext[0]:.3f} Y={ext[1]:.3f} Z={ext[2]:.3f}  "
          f"palm-normal(thinnest)={'XYZ'[int(ext.argmin())]}")
    return ext


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand", type=str, default="wuji")
    p.add_argument("--human", type=str, default="mano_right")
    p.add_argument("--n", type=int, default=20000)
    args = p.parse_args()

    config = get_config(args.hand)
    print(f"=== ROBOT ({args.hand}) fingertip cloud in base_link frame ===")
    robot_pts, names = robot_tip_cloud(config, n=args.n)
    robot_flat = np.concatenate([robot_pts[n] for n in names], axis=0)
    r_ext = axis_report("robot all", robot_flat)
    # robot thumb-index overlap
    if "thumb" in robot_pts and "index" in robot_pts:
        # independent pairing (matches how training treats it)
        ti = np.linalg.norm(robot_pts["thumb"][:len(robot_pts["index"])] - robot_pts["index"], axis=1)
        print(f"  robot thumb-index min dist {ti.min():.3f}m, <1.5cm {(ti<0.015).sum()}/{len(ti)} (overlap => pinch trainable)")

    print(f"\n=== HUMAN ({args.human}) fingertip cloud (palm frame) ===")
    hp = np.load(get_human_data(args.human))  # [T,21,3]
    human_tips = hp[:, MP_TIP_IDX, :].reshape(-1, 3)
    h_ext = axis_report("human all", human_tips)

    # Robust criterion: the palm normal is the thinnest extent for both clouds, and
    # the thumb should sit at +Y, pinky at -Y. Y/Z dominance can swap (wide thumb
    # abduction inflates robot Y), so do NOT compare the *longest* axis.
    robot_thumb_y = robot_pts[names[0]].mean(0)[1]
    robot_pinky_y = robot_pts[names[-1]].mean(0)[1]
    human_thumb_y = hp[:, 4].mean(0)[1]
    human_pinky_y = hp[:, 20].mean(0)[1]
    print("\n=== verdict ===")
    normal_ok = r_ext.argmin() == h_ext.argmin()
    thumb_ok = (robot_thumb_y > robot_pinky_y) == (human_thumb_y > human_pinky_y)
    print(f"  palm-normal axis: robot='{'XYZ'[int(r_ext.argmin())]}' human='{'XYZ'[int(h_ext.argmin())]}' -> {'AGREE' if normal_ok else 'MISMATCH'}")
    print(f"  thumb-vs-pinky along +Y: robot {'thumb+' if robot_thumb_y>robot_pinky_y else 'pinky+'}, "
          f"human {'thumb+' if human_thumb_y>human_pinky_y else 'pinky+'} -> {'AGREE' if thumb_ok else 'MISMATCH'}")
    if normal_ok and thumb_ok:
        print("  Frames are compatible. Proceed to training (Chamfer aligns distributions).")
    else:
        print("  Frames disagree. Add a virtual base link (fixed joint + rotation) to the")
        print("  wuji URDF so +Z runs along the fingers, +Y toward the thumb, +X out of the")
        print("  palm; point config 'base_link' at it and re-run this check.")


if __name__ == "__main__":
    main()
