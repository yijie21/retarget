# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Process Arctic raw_seqs into SPIDER's mjwp_fast schema.

This pipeline reads Arctic-v2 ``raw_seqs/<subject>/<sequence>.{mano,object}.npy``
plus ``meta/object_vtemplates/<obj>/bottom.obj`` and produces a single
clip per (subject, sequence). Specifically:

1. Filters to the 6 dexmachina-pinned objects
   (box, ketchup, laptop, mixer, notebook, waffleiron). See
   :mod:`spider.postprocess.evaluate_dexmachina` lines 230-234.
2. Uses ONLY the object's bottom part (``bottom.obj``). The object is
   treated as a single rigid body; no articulation. The wrist contacts
   the bottom (the part that the hand picks up).
3. Clips to the picking stage only: detects when the object first starts
   moving (translation away from its frame-0 position), keeps
   ``pre_grasp_seconds`` (default 2s) before that and ``post_lift_seconds``
   (default 2s) after. This matches the user's "object-motion start"
   criterion.
4. Floor offset: shifts the entire scene (object + both hands) so that the
   object's lowest world-z at the FIRST CLIP FRAME sits at z=0. Arctic
   captures place the table well above z=0; without this shift the floor
   in the simulator scene is below the object.
5. Bimanual: both hand poses come from MANO FK; ``active_hand`` records
   which hand(s) actually contact the object.

Author: Chaoyi Pan
Date: 2026-05-31
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import loguru
import mujoco
import numpy as np
import pymeshlab
import torch
import trimesh
import tyro
from scipy.spatial.transform import Rotation as R

import spider
from spider.io import get_mesh_dir, get_processed_data_dir

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Arctic mocap rate
FPS_MOCAP = 30
# manotorch MANO 21-joint indices for the 5 fingertips (SNAP layout)
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
# bone connectivity for the 21-joint skeleton (parent -> child)
SKELETON_BONES = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# 6 dexmachina-pinned objects.
# See spider/postprocess/evaluate_dexmachina.py:230-234 for the canonical
# enumeration; not exported there, so duplicated here.
ARCTIC_OBJECTS = ["box", "ketchup", "laptop", "mixer", "notebook", "waffleiron"]


def _ensure_mano_assets_layout(mano_assets_root: str) -> str:
    """Return a directory ``D`` such that ``D/models/MANO_RIGHT.pkl`` exists.

    manotorch.ManoLayer expects ``mano_assets_root/models/MANO_<side>.pkl``.
    Arctic ships ``body_models/mano/MANO_<side>.pkl`` and a separate
    ``body_models/models/smplx/`` folder. If the user passed
    ``~/arctic/unpack/body_models``, symlink ``mano/MANO_*.pkl`` into
    ``models/`` so manotorch can find them; if the user passed
    ``~/arctic/unpack/body_models/mano`` directly, create a sibling
    ``models/`` symlink target.
    """
    mano_assets_root = os.path.abspath(os.path.expanduser(mano_assets_root))
    target = os.path.join(mano_assets_root, "models")
    os.makedirs(target, exist_ok=True)
    # Candidates where the .pkl might actually live
    candidates_dirs = [
        mano_assets_root,
        os.path.join(mano_assets_root, "mano"),
        os.path.dirname(mano_assets_root),  # parent
        os.path.join(os.path.dirname(mano_assets_root), "mano"),
    ]
    for side in ("RIGHT", "LEFT"):
        dst = os.path.join(target, f"MANO_{side}.pkl")
        if os.path.exists(dst):
            continue
        # remove a broken symlink left from a previous run before re-creating
        if os.path.lexists(dst):
            os.unlink(dst)
        src = None
        for d in candidates_dirs:
            cand = os.path.join(d, f"MANO_{side}.pkl")
            if os.path.exists(cand):
                src = cand
                break
        if src is None:
            raise FileNotFoundError(
                f"Could not find MANO_{side}.pkl under any of: {candidates_dirs}. "
                "Pass --mano-assets-root pointing at a directory whose "
                "siblings/children contain MANO_RIGHT.pkl and MANO_LEFT.pkl."
            )
        os.symlink(src, dst)
    return mano_assets_root


def _patch_chumpy_compat() -> None:
    """Make chumpy 0.70 importable on Python 3.12 + NumPy 2.x.

    chumpy (manotorch transitive dep) does ``from numpy import bool, int, ...``
    and ``inspect.getargspec``, both removed in modern stacks. Inject the
    legacy attributes so chumpy can import; we never use those classes
    ourselves.
    """
    import inspect as _inspect

    import numpy as _np

    if not hasattr(_inspect, "getargspec"):
        _inspect.getargspec = _inspect.getfullargspec  # type: ignore[attr-defined]
    for alias, impl in [
        ("bool", bool), ("int", int), ("float", float), ("complex", complex),
        ("object", object), ("unicode", str), ("str", str),
    ]:
        if not hasattr(_np, alias):
            setattr(_np, alias, impl)


def _load_mano_layers(mano_assets_root: str) -> tuple:
    """Return (right_layer, left_layer) ManoLayer instances.

    Arctic raw_seqs MANO parameters are stored relative to the **non-flat**
    MANO mean pose (see ``arctic/common/body_models.py``: ``build_mano_aa``
    is called with ``flat_hand=False``). manotorch must use the same to
    reproduce Arctic's joints — using ``flat_hand_mean=True`` would offset
    every finger joint by the MANO mean.

    ``center_idx=None`` (NOT ``0``) so the wrist joint isn't shifted to the
    origin before adding ``trans``. With ``center_idx=0`` manotorch outputs
    ``J_canonical_zeroed_at_wrist + trans``, while ``smplx.MANO`` (what
    Arctic uses) outputs ``J_canonical + trans`` — using ``center_idx=0``
    moves the wrist by the canonical wrist offset (~5-10 cm, mostly in x)
    and shifts every fingertip with it.
    """
    _patch_chumpy_compat()
    from manotorch.manolayer import ManoLayer

    root = _ensure_mano_assets_layout(mano_assets_root)
    right = ManoLayer(
        mano_assets_root=root,
        rot_mode="axisang",
        side="right",
        center_idx=None,
        use_pca=False,
        flat_hand_mean=False,
    )
    left = ManoLayer(
        mano_assets_root=root,
        rot_mode="axisang",
        side="left",
        center_idx=None,
        use_pca=False,
        flat_hand_mean=False,
    )
    return right, left


def _mano_fk(
    mano_layer,
    rot: np.ndarray,    # (T, 3) global orient axis-angle
    pose: np.ndarray,   # (T, 45) finger axis-angle
    trans: np.ndarray,  # (T, 3) wrist translation, meters
    shape: np.ndarray,  # (10,) betas
) -> tuple[np.ndarray, np.ndarray]:
    """Run MANO FK with arctic params.

    Returns (joints_world (T,21,3), verts_world (T, 778, 3)).
    """
    T = rot.shape[0]
    pose_coeffs = torch.from_numpy(
        np.concatenate([rot, pose], axis=-1)
    ).float()  # (T, 48)
    betas = torch.from_numpy(np.tile(shape[None, :], (T, 1))).float()
    tsl = torch.from_numpy(trans).float()
    with torch.no_grad():
        out = mano_layer(pose_coeffs=pose_coeffs, betas=betas)
        joints = out.joints + tsl[:, None, :]
        verts = out.verts + tsl[:, None, :]
    return joints.cpu().numpy(), verts.cpu().numpy()


def _build_bottom_only_object_mesh(
    obj_template_dir: str,
    out_path: str,
) -> None:
    """Export Arctic's bottom-part mesh as the rigid object mesh.

    Arctic's object is articulated (top + bottom around a hinge). Per the
    requested workflow, we keep only the bottom part — that is the body the
    hand grasps and lifts. ``bottom.obj`` is shipped under
    ``meta/object_vtemplates/<obj>/`` in millimetres; we convert to metres.
    """
    mesh_path = os.path.join(obj_template_dir, "bottom.obj")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(
            f"Expected bottom.obj under {obj_template_dir}; not found."
        )
    mesh = trimesh.load(mesh_path, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64) / 1000.0  # mm -> m
    faces = np.asarray(mesh.faces, dtype=np.int64)
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out.export(out_path)


def _detect_grasp_window(
    obj_trans_m: np.ndarray,       # (T, 3) bottom-part world translation, metres
    rh_verts_world: np.ndarray,    # (T, 778, 3)
    lh_verts_world: np.ndarray,    # (T, 778, 3)
    obj_verts_world: np.ndarray,   # (T, V_obj_sub, 3)  for active-hand assignment
    fps: float,
    pre_grasp_seconds: float,
    post_lift_seconds: float,
    motion_thresh_m: float = 0.02,
    contact_thresh: float = 0.05,
) -> tuple[int, int, str]:
    """Detect the picking window using object motion onset.

    The "grasp moment" is when the object first begins to move significantly
    away from its frame-0 translation. We center the clip on that moment:

        start = motion_onset - pre_grasp_seconds * fps
        end   = motion_onset + post_lift_seconds * fps

    The active hand is whichever hand's MANO verts come closest to the
    object around the motion-onset frame.

    Args:
        obj_trans_m: per-frame object translation (bottom-part centre) in
            metres; motion onset uses ``||trans - trans[0]|| > motion_thresh_m``.
        rh_verts_world / lh_verts_world: MANO vertices for active-hand
            classification.
        obj_verts_world: subsampled object verts for hand-distance check.
        motion_thresh_m: how far (in metres) the object must translate
            from its frame-0 position to count as "moving".
    """
    T = obj_trans_m.shape[0]
    disp = np.linalg.norm(obj_trans_m - obj_trans_m[0:1], axis=-1)
    moving_frames = np.where(disp > motion_thresh_m)[0]
    if moving_frames.size == 0:
        # Object never moves — fall back to the frame of largest displacement.
        motion_onset = int(np.argmax(disp))
    else:
        motion_onset = int(moving_frames[0])

    # Active-hand: whichever hand is closest to the object near onset.
    rh_sub = rh_verts_world[:, ::8, :]
    lh_sub = lh_verts_world[:, ::8, :]
    win_lo = max(0, motion_onset - int(round(0.25 * fps)))
    win_hi = min(T, motion_onset + int(round(0.25 * fps)) + 1)
    rh_d = []
    lh_d = []
    for t in range(win_lo, win_hi):
        rh_d.append(np.linalg.norm(
            rh_sub[t][:, None, :] - obj_verts_world[t][None, :, :], axis=-1
        ).min())
        lh_d.append(np.linalg.norm(
            lh_sub[t][:, None, :] - obj_verts_world[t][None, :, :], axis=-1
        ).min())
    rh_min = float(min(rh_d)) if rh_d else 1e9
    lh_min = float(min(lh_d)) if lh_d else 1e9
    if rh_min < contact_thresh and lh_min < contact_thresh:
        active = "both"
    elif rh_min < contact_thresh:
        active = "right"
    elif lh_min < contact_thresh:
        active = "left"
    else:
        active = "both"

    pre_n = int(round(pre_grasp_seconds * fps))
    post_n = int(round(post_lift_seconds * fps))
    start = max(0, motion_onset - pre_n)
    end = min(T, motion_onset + post_n + 1)
    if end <= start:
        end = min(T, start + 1)
    return start, end, active


def main(
    arctic_root: Path = Path("~/arctic"),  # noqa: B008  expanded inside main
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    subject: str = "s01",
    sequence: str = "box_use_01",
    data_id: int = 0,
    embodiment_type: str = "bimanual",
    robot_type: str = "xhand",
    pre_grasp_seconds: float = 2.0,
    post_lift_seconds: float = 2.0,
    motion_thresh_m: float = 0.02,
    target_fps: float = 30.0,
    mano_assets_root: str | None = None,
    no_preview: bool = False,
    show_viewer: bool = True,
):
    """Process a single (subject, sequence) Arctic clip.

    Args:
        arctic_root: Root of the unpacked Arctic dataset (containing
            ``unpack/arctic_data/data/raw_seqs`` and ``unpack/body_models``).
        dataset_dir: Where to write processed outputs.
        subject: ``s01`` ... ``s10``.
        sequence: e.g. ``box_use_01``. Object name is the prefix before the
            first underscore.
        data_id: Output sub-directory.
        embodiment_type: Always ``bimanual`` for Arctic.
        robot_type: e.g. ``xhand``, ``sharpa``. Used only to name the output
            dir; the schema itself is robot-agnostic.
        pre_grasp_seconds: Seconds before object-motion onset to keep.
        post_lift_seconds: Seconds after object-motion onset to keep.
        motion_thresh_m: Object is considered "moving" once its translation
            has moved this far (metres) from its frame-0 position.
        target_fps: Output FPS. Arctic raw is 30 Hz; default keeps it at 30.
        mano_assets_root: Directory containing MANO pickles. On this machine
            it lives under ``~/arctic/unpack/body_models``. If not provided
            we auto-detect from a few common locations under ``arctic_root``.
        no_preview: Alias for ``--no-show-viewer``.
        show_viewer: Open the viser preview after writing the NPZ.
    """
    if no_preview:
        show_viewer = False
    if embodiment_type != "bimanual":
        loguru.logger.warning(
            f"arcticv2 is bimanual; got embodiment_type={embodiment_type}. "
            "Proceeding but downstream consumers expect 'bimanual'."
        )

    arctic_root = Path(os.path.abspath(os.path.expanduser(str(arctic_root))))
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    if mano_assets_root is None:
        def _has_mano(d: Path) -> bool:
            return any(
                (d / sub / f"MANO_{side}.pkl").exists()
                for side in ("RIGHT", "LEFT")
                for sub in ("", "mano", "models", "mano/models")
            )

        for c in (
            arctic_root / "unpack" / "body_models",
            arctic_root / "body_models",
            arctic_root / "unpack" / "body_models" / "mano",
            arctic_root / "body_models" / "mano",
        ):
            if c.exists() and _has_mano(c):
                mano_assets_root = str(c)
                break
        else:
            mano_assets_root = str(arctic_root / "unpack" / "body_models")

    # Object-class filter (the 6 dexmachina objects).
    obj_name = sequence.split("_")[0]
    if obj_name not in ARCTIC_OBJECTS:
        raise ValueError(
            f"Sequence '{sequence}' object='{obj_name}' is not in the "
            f"dexmachina-pinned set {ARCTIC_OBJECTS}."
        )

    # Articulation discarded warning.
    loguru.logger.warning(
        "arcticv2: keeping ONLY the object's bottom part (the lifted body); "
        "the articulated top half and articulation angle are discarded."
    )

    # 1. Load raw arrays.
    seq_dir = arctic_root / "unpack/arctic_data/data/raw_seqs" / subject
    mano_path = seq_dir / f"{sequence}.mano.npy"
    object_path = seq_dir / f"{sequence}.object.npy"
    if not mano_path.exists():
        raise FileNotFoundError(f"Missing {mano_path}")
    if not object_path.exists():
        raise FileNotFoundError(f"Missing {object_path}")
    mano = np.load(mano_path, allow_pickle=True).item()
    obj = np.load(object_path, allow_pickle=True)  # (T, 7) float32
    if obj.ndim != 2 or obj.shape[-1] != 7:
        raise ValueError(f"Unexpected object array shape: {obj.shape}")

    # Object: [arti_rad, axang_x, axang_y, axang_z, tx_mm, ty_mm, tz_mm]
    obj_arti = obj[:, 0].astype(np.float32)
    obj_axang = obj[:, 1:4].astype(np.float32)
    obj_trans_m = (obj[:, 4:7].astype(np.float32)) / 1000.0  # mm -> m

    T = obj.shape[0]
    if mano["right"]["rot"].shape[0] != T:
        raise ValueError(
            f"frame-count mismatch: object T={T}, mano T={mano['right']['rot'].shape[0]}"
        )
    loguru.logger.info(
        f"Loaded {sequence} ({subject}): T={T} frames @ {FPS_MOCAP}Hz, "
        f"object={obj_name}"
    )

    obj_template_dir = (
        arctic_root / "unpack/arctic_data/data/meta/object_vtemplates" / obj_name
    )

    # 2. MANO FK on the FULL trajectory (so the grasp detector sees real
    # vertex distances). Both hands.
    rh_layer, lh_layer = _load_mano_layers(mano_assets_root)
    rh_joints, rh_verts = _mano_fk(
        rh_layer, mano["right"]["rot"], mano["right"]["pose"],
        mano["right"]["trans"], mano["right"]["shape"],
    )
    lh_joints, lh_verts = _mano_fk(
        lh_layer, mano["left"]["rot"], mano["left"]["pose"],
        mano["left"]["trans"], mano["left"]["shape"],
    )

    # 3. Build per-frame bottom-part world verts (used for the active-hand
    # check). Object pose in raw_seqs/<seq>.object.npy gives the bottom
    # part's world rotation+translation directly.
    bottom_mesh = trimesh.load(
        str(obj_template_dir / "bottom.obj"), process=False,
    )
    bottom_verts_local = np.asarray(bottom_mesh.vertices, dtype=np.float64) / 1000.0
    sub = bottom_verts_local[:: max(1, bottom_verts_local.shape[0] // 500)]
    obj_world_verts = np.empty((T, sub.shape[0], 3), dtype=np.float64)
    for t in range(T):
        Rt = R.from_rotvec(obj_axang[t])
        obj_world_verts[t] = Rt.apply(sub) + obj_trans_m[t]

    # 4. Detect picking window from object-motion onset.
    start, end, active_hand = _detect_grasp_window(
        obj_trans_m=obj_trans_m,
        rh_verts_world=rh_verts,
        lh_verts_world=lh_verts,
        obj_verts_world=obj_world_verts,
        fps=FPS_MOCAP,
        pre_grasp_seconds=pre_grasp_seconds,
        post_lift_seconds=post_lift_seconds,
        motion_thresh_m=motion_thresh_m,
    )
    loguru.logger.info(
        f"clipped frame range = [{start}, {end}) ({end - start} frames @ "
        f"{FPS_MOCAP}Hz); active_hand={active_hand}"
    )

    # 5. Export the bottom-only object mesh.
    safe_name = f"{obj_name}_bottom"
    mesh_dir = get_mesh_dir(
        dataset_dir=dataset_dir, dataset_name="arcticv2", object_name=safe_name,
    )
    os.makedirs(mesh_dir, exist_ok=True)
    out_mesh = os.path.join(mesh_dir, "visual.obj")
    _build_bottom_only_object_mesh(str(obj_template_dir), out_mesh)
    loguru.logger.info(f"Wrote bottom-only object mesh to {out_mesh}")

    # 6. Slice all per-frame arrays to [start, end).
    rh_joints = rh_joints[start:end]
    rh_verts = rh_verts[start:end]
    lh_joints = lh_joints[start:end]
    lh_verts = lh_verts[start:end]
    obj_axang_clip = obj_axang[start:end]
    obj_trans_clip = obj_trans_m[start:end]
    rh_rot_clip = mano["right"]["rot"][start:end]
    lh_rot_clip = mano["left"]["rot"][start:end]

    # 7. Optional resample. Arctic is 30 fps; default target_fps=30 -> no-op.
    n_in = end - start
    if abs(target_fps - FPS_MOCAP) > 1e-3:
        n_out = int(round(n_in * target_fps / FPS_MOCAP))
        idx = np.linspace(0, n_in - 1, num=max(2, n_out)).round().astype(int)
        rh_joints = rh_joints[idx]
        rh_verts = rh_verts[idx]
        lh_joints = lh_joints[idx]
        lh_verts = lh_verts[idx]
        obj_axang_clip = obj_axang_clip[idx]
        obj_trans_clip = obj_trans_clip[idx]
        rh_rot_clip = rh_rot_clip[idx]
        lh_rot_clip = lh_rot_clip[idx]
        n = len(idx)
    else:
        n = n_in
    loguru.logger.info(f"output {n} frames @ {target_fps}Hz")

    # 8. Output dirs.
    task = f"{subject}-{sequence}"
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name="arcticv2",
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(output_dir, exist_ok=True)

    # 9. Coordinate frame: Arctic captures are already z-up in metres
    # (top.obj + obj_trans/1000 + MANO trans share the same world). No
    # global rotation is needed.
    unit_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # Wrist offsets: map MANO canonical hand frame to xhand's per-side
    # ``{right,left}_palm`` site frame so the palm faces the correct side
    # and fingers extend in the same direction as MANO's TPose.
    # See spider/assets/robots/xhand/{right,left}.xml for site quats; columns
    # of these matrices are (site_x, site_y, site_z) expressed in the MANO
    # canonical-hand basis (right: fingers→-x, thumb→+z; left: fingers→+x,
    # thumb→+z).
    R_offset_right = R.from_matrix(np.array([
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]))
    R_offset_left = R.from_matrix(np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]))

    qpos_wrist_right = np.zeros((n, 7))
    qpos_finger_right = np.zeros((n, 5, 7))
    qpos_obj_right = np.zeros((n, 7))
    qpos_wrist_left = np.zeros((n, 7))
    qpos_finger_left = np.zeros((n, 5, 7))
    qpos_obj_left = np.zeros((n, 7))

    # Per-hand wrist + fingertips from the FK output (joint 0 = wrist).
    for i in range(n):
        qpos_wrist_right[i, :3] = rh_joints[i, 0]
        r = R.from_rotvec(rh_rot_clip[i]) * R_offset_right
        xyzw = r.as_quat()
        qpos_wrist_right[i, 3:] = np.concatenate([xyzw[3:], xyzw[:3]])  # wxyz

        qpos_wrist_left[i, :3] = lh_joints[i, 0]
        r = R.from_rotvec(lh_rot_clip[i]) * R_offset_left
        xyzw = r.as_quat()
        qpos_wrist_left[i, 3:] = np.concatenate([xyzw[3:], xyzw[:3]])

        for j in range(5):
            qpos_finger_right[i, j, :3] = rh_joints[i, FINGERTIP_INDICES[j]]
            qpos_finger_right[i, j, 3:] = unit_quat
            qpos_finger_left[i, j, :3] = lh_joints[i, FINGERTIP_INDICES[j]]
            qpos_finger_left[i, j, 3:] = unit_quat

    # Object pose is the bottom link's world pose. The Arctic scene has only
    # ONE object (both hands manipulate it together); ``qpos_obj_left`` is a
    # zero-padded placeholder so the schema matches the oakinkv2-style empty
    # left-object body in the generated scene XML.
    for i in range(n):
        r = R.from_rotvec(obj_axang_clip[i])
        xyzw = r.as_quat()
        wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
        qpos_obj_right[i, :3] = obj_trans_clip[i]
        qpos_obj_right[i, 3:] = wxyz
    qpos_obj_left[:, :3] = 0.0
    qpos_obj_left[:, 3:] = unit_quat

    # 10. Floor offset: shift the entire scene so the bottom-part object's
    # lowest world-z at the FIRST CLIP FRAME sits at z=0. Arctic captures
    # have the table well above z=0; without this shift the simulator floor
    # would sit far below the object and the floor-contact check is wrong.
    visual_mesh = pymeshlab.MeshSet()
    visual_mesh.load_new_mesh(out_mesh)
    obj_verts_local_full = np.asarray(visual_mesh.current_mesh().vertex_matrix())
    R_obj0 = R.from_rotvec(obj_axang_clip[0])
    obj_lowest_z_pre = float(
        (R_obj0.apply(obj_verts_local_full) + qpos_obj_right[0, :3]).min(axis=0)[2]
    )
    z_offset = -obj_lowest_z_pre
    loguru.logger.info(
        f"Floor offset: shifting scene by dz={z_offset:+.3f} m so the "
        f"object's lowest z at frame 0 (was {obj_lowest_z_pre:+.3f} m) "
        f"becomes 0."
    )
    qpos_wrist_right[:, 2] += z_offset
    qpos_wrist_left[:, 2] += z_offset
    qpos_finger_right[:, :, 2] += z_offset
    qpos_finger_left[:, :, 2] += z_offset
    qpos_obj_right[:, 2] += z_offset
    # qpos_obj_left is a zero placeholder (empty left-object body); leave it.
    # Keep the FK joint arrays aligned for the viser preview.
    rh_joints[..., 2] += z_offset
    lh_joints[..., 2] += z_offset

    obj_min_z_per_frame = []
    for k in range(n):
        Rk = R.from_rotvec(obj_axang_clip[k])
        zk = (Rk.apply(obj_verts_local_full) + qpos_obj_right[k, :3]).min(axis=0)[2]
        obj_min_z_per_frame.append(zk)
    obj_min_z_traj = float(np.min(obj_min_z_per_frame))
    obj_first_frame_lowest_z = float(
        (R_obj0.apply(obj_verts_local_full) + qpos_obj_right[0, :3]).min(axis=0)[2]
    )
    scene_min_z = float(min(
        obj_min_z_traj,
        qpos_wrist_right[:, 2].min(),
        qpos_finger_right[:, :, 2].min(),
        qpos_wrist_left[:, 2].min(),
        qpos_finger_left[:, :, 2].min(),
    ))
    object_descends = obj_min_z_traj < obj_first_frame_lowest_z - 0.02

    # 11. task_info.
    rel_mesh_dir = str(Path(mesh_dir).relative_to(dataset_dir))
    task_info = {
        "task": task,
        "dataset_name": "arcticv2",
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        # The Arctic scene has ONE object that both hands manipulate. Emit it
        # only on the right side so generate_xml doesn't build two stacked
        # free-floating bodies at the same pose. ``qpos_obj_left`` is still
        # written to the NPZ as a zero-padding so downstream loaders that
        # always expect both arrays continue to work; the left object body
        # in the scene XML will be empty (gravcomp=1) like in oakinkv2.
        "right_object_mesh_dir": rel_mesh_dir,
        "left_object_mesh_dir": None,
        "ref_dt": 1.0 / target_fps,
        "n_frames": n,
        "obj_first_frame_lowest_world_z": obj_first_frame_lowest_z,
        "scene_lowest_world_z": scene_min_z,
        "object_descends_from_frame0": bool(object_descends),
        "z_offset_applied": z_offset,
        # arctic-specific provenance
        "arctic_subject": subject,
        "arctic_object": obj_name,
        "arctic_clip_use": sequence,
        "arctic_frame_range": [int(start), int(end)],
        "active_hand": active_hand,
        "pre_grasp_seconds": pre_grasp_seconds,
        "post_lift_seconds": post_lift_seconds,
        "motion_thresh_m": motion_thresh_m,
        "articulation_discarded": True,
    }
    task_info_path = f"{output_dir}/../task_info.json"
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Saved task_info to {task_info_path}")

    np.savez(
        f"{output_dir}/trajectory_keypoints.npz",
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
    )
    loguru.logger.info(f"Saved qpos to {output_dir}/trajectory_keypoints.npz")
    loguru.logger.info(
        f"After offset: obj frame-0 lowest world z = {obj_first_frame_lowest_z:+.3f} m; "
        f"scene lowest z = {scene_min_z:+.3f} m"
        + (" (object descends from frame 0)" if object_descends else " (object stays above frame-0 z)")
    )

    if not show_viewer:
        return

    # 12. Optional viser preview: build a minimal mocap scene with the rigid
    # object mesh and a per-hand 21-joint skeleton overlay.
    mj_spec = mujoco.MjSpec.from_file(f"{spider.ROOT}/assets/mano/empty_scene.xml")
    object_right_handle = mj_spec.worldbody.add_body(name="right_object", mocap=True)
    object_right_handle.add_site(
        name="right_object", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.001, 0.001, 0.001], rgba=[1, 0, 0, 0], group=3,
    )
    mj_spec.add_mesh(name="right_object", file=out_mesh)
    object_right_handle.add_geom(
        name="right_object", type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="right_object", pos=[0, 0, 0], quat=[1, 0, 0, 0],
        group=0, condim=1,
    )
    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)

    qpos_list = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
        ],
        axis=1,
    )

    from spider.viewers.viser_viewer import (
        _STATE,
        build_and_log_scene_from_spec,
        init_viser,
        log_frame,
    )

    init_viser(app_name="spider-arcticv2")
    body_entity_and_ids = build_and_log_scene_from_spec(
        mj_spec, mj_model, build_ref=False
    )
    server = _STATE.server

    rh_handles = [
        server.scene.add_icosphere(
            f"hand/right/joint_{i}", radius=0.006,
            color=(220, 80, 80) if i in FINGERTIP_INDICES else (80, 130, 220),
            position=tuple(rh_joints[0, i]),
        )
        for i in range(21)
    ]
    lh_handles = [
        server.scene.add_icosphere(
            f"hand/left/joint_{i}", radius=0.006,
            color=(220, 140, 80) if i in FINGERTIP_INDICES else (130, 200, 80),
            position=tuple(lh_joints[0, i]),
        )
        for i in range(21)
    ]
    bone_handles = []
    for parent, child in SKELETON_BONES:
        h_r = server.scene.add_spline_catmull_rom(
            f"hand/right/bone_{parent}_{child}",
            positions=np.stack([rh_joints[0, parent], rh_joints[0, child]]),
            color=(255, 200, 50), line_width=2.5,
        )
        h_l = server.scene.add_spline_catmull_rom(
            f"hand/left/bone_{parent}_{child}",
            positions=np.stack([lh_joints[0, parent], lh_joints[0, child]]),
            color=(180, 255, 80), line_width=2.5,
        )
        bone_handles.append((h_r, h_l, parent, child))

    loguru.logger.info(
        f"Viser scene built; pushing {n} frames. Open the printed URL "
        "in your browser."
    )
    for t in range(n):
        mj_data.mocap_pos[:] = qpos_list[t, :, :3]
        mj_data.mocap_quat[:] = qpos_list[t, :, 3:]
        mujoco.mj_kinematics(mj_model, mj_data)
        log_frame(
            mj_data, sim_time=t / target_fps,
            viewer_body_entity_and_ids=body_entity_and_ids,
            show_ui=True, playback_fps=target_fps,
        )

    def _update_hand(frame_idx: int) -> None:
        frame_idx = max(0, min(frame_idx, n - 1))
        for i, h in enumerate(rh_handles):
            h.position = tuple(rh_joints[frame_idx, i])
        for i, h in enumerate(lh_handles):
            h.position = tuple(lh_joints[frame_idx, i])
        for h_r, h_l, parent, child in bone_handles:
            h_r.positions = np.stack([
                rh_joints[frame_idx, parent], rh_joints[frame_idx, child]
            ])
            h_l.positions = np.stack([
                lh_joints[frame_idx, parent], lh_joints[frame_idx, child]
            ])

    if _STATE.playback_slider is not None:
        @_STATE.playback_slider.on_update
        def _(_):
            _update_hand(int(_STATE.playback_slider.value))

    loguru.logger.info("Press Ctrl-C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
