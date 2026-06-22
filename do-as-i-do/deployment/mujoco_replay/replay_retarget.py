"""Replay a spider mjwp retarget trajectory on the dual_ur3e scene (mink IK).

Drives one arm + its Sharpa hand (--side right [default] or left). The
retargeted Sharpa wrist pose lives in spider's world frame; per frame we

  1. decode the desired Sharpa-root world pose from qpos[0:6]
     (3 slide joints + 3 hinge joints = xyz + intrinsic XYZ-Euler),
  2. apply a configurable workspace transform (translate + yaw/pitch/roll
     around the start-frame anchor) to land it in the dual_ur3e world,
  3. invert the flange→sharparoot mount used in build_scene.py to get
     the target pose for the UR3e attachment_site,
  4. solve IK on the chosen UR3e (6 DOF): a mink QP drives a FrameTask on the
     attachment_site while a CollisionAvoidanceLimit keeps the arm from
     intersecting itself, the table/breadboard/blocks and (optionally) the
     other arm; a DofFreezingTask pins every non-arm dof so only the 6 arm
     joints move.
  5. copy the 22 finger joints straight through.

The other arm and hand stay at the home keyframe.

Usage:

python dual_ur3e/replay_retarget.py --side left --traj .../trajectory_mjwp.npz

# headless: solve + print residuals/collision stats, no viser GUI
python dual_ur3e/replay_retarget.py --side left --traj .../trajectory_mjwp.npz --solve-only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mink
import mujoco
import numpy as np
import viser

from build_scene import build

# Mirrors the HAND_OFFSET constant inside build_scene.build():
# coupler_black_height (0.01325) + coupler_silver_height (0.0175) m
# along the attachment_site +Z axis.
HAND_OFFSET = 0.03075
from mjviser import ViserMujocoScene

# Placeholder default — pass --traj explicitly to point at a spider
# trajectory_mjwp.npz produced by the retargeting stage.
DEFAULT_TRAJ = Path("trajectory_mjwp.npz")

# Sharpa-root yaw in the flange frame, matching build_scene.z_quat(...) for
# each hand: right_hand_C_MC = z_quat(135), left_hand_C_MC = z_quat(45).
HAND_MOUNT_YAW_DEG = {"right": 135.0, "left": 45.0}

# Saved preset: world position where the wrist (sharpa root) sits at the
# start_frame. In dual_ur3e world coords: +x toward far wall, +y toward
# the left arm, +z up. Each side's default lands the wrist in front of that
# arm at a comfortable working height. Pass via --workspace-x/-y/-z to
# override. The left preset mirrors the right across the y=center plane
# (right arm sits at -y, left arm at +y) — it's only a starting guess; tune
# yaw/pitch/roll in the GUI.
PRESET_FRONT_OF_ARM = {
    "right": {
        "x": -0.29, "y": -0.16, "z": 0.89, "yaw": 90.0, "pitch": 0.0, "roll": 0.0,
    },
    "left": {
        "x": -0.29, "y": 0.16, "z": 0.89, "yaw": -90.0, "pitch": 0.0, "roll": 0.0,
    },
}
DEFAULT_START_FRAME = 600

# Filename written by the GUI "Save retarget" button — also the name we
# look for when auto-detecting a reference file next to the input traj.
RETARGET_FILENAME = "trajectory_dual_ur3e.npz"


def _load_reference_defaults(path: Path) -> dict | None:
    """Read workspace_xyz / workspace_yaw_deg / start_frame from a saved
    retarget npz. Returns a dict with keys x, y, z, yaw, pitch, roll,
    start_frame, or None if the file can't be parsed.
    """
    try:
        with np.load(path) as npz:
            xyz = np.asarray(npz["workspace_xyz"], dtype=np.float64)
            yaw = float(np.asarray(npz["workspace_yaw_deg"]))
            start_frame = int(np.asarray(npz["start_frame"]))
            # pitch/roll were added after yaw; older retargets default to 0.
            pitch = (
                float(np.asarray(npz["workspace_pitch_deg"]))
                if "workspace_pitch_deg" in npz.files else 0.0
            )
            roll = (
                float(np.asarray(npz["workspace_roll_deg"]))
                if "workspace_roll_deg" in npz.files else 0.0
            )
    except (OSError, KeyError, ValueError) as e:
        print(f"  warning: could not read reference {path}: {e}")
        return None
    if xyz.shape != (3,):
        print(f"  warning: reference {path} has bad workspace_xyz shape {xyz.shape}")
        return None
    return {
        "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]),
        "yaw": yaw, "pitch": pitch, "roll": roll, "start_frame": start_frame,
    }


# Spider hand qpos layout (per frame, total 35):
#   [0:3]  wrist xyz (slide joints)
#   [3:6]  wrist rotation (hinges Rx, Ry, Rz applied as nested bodies)
#   [6:28] 22 finger joints
#   [28:35] object free joint (ignored here)
SPIDER_FINGER_SLICE = slice(6, 28)


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a wxyz quaternion from axis-angle."""
    out = np.zeros(4)
    mujoco.mju_axisAngle2Quat(out, axis, angle)
    return out


_X_AXIS = np.array([1.0, 0.0, 0.0])
_Y_AXIS = np.array([0.0, 1.0, 0.0])
_Z_AXIS = np.array([0.0, 0.0, 1.0])


def _spider_wrist_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """Compose the spider wrist hinge chain into a single wxyz quat.

    Bodies are nested base_roll(Rx) → base_pitch(Ry) → base_yaw(Rz) →
    <side>_hand_C_MC, so the root's rotation in world is Rx · Ry · Rz.
    """
    qx = _axis_angle_quat(_X_AXIS, rx)
    qy = _axis_angle_quat(_Y_AXIS, ry)
    qz = _axis_angle_quat(_Z_AXIS, rz)
    return _quat_mul(qx, _quat_mul(qy, qz))


def _spider_root_pose(qpos_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = qpos_frame[0:3].copy()
    quat = _spider_wrist_quat(
        float(qpos_frame[3]), float(qpos_frame[4]), float(qpos_frame[5])
    )
    return pos, quat


def _flange_offset_quat(mount_yaw_deg: float) -> np.ndarray:
    """Quaternion (wxyz) of the Sharpa root in the flange frame.

    build_scene mounts <side>_hand_C_MC at z_quat(mount_yaw_deg) relative to
    the attachment_site (135° right, 45° left).
    """
    return _axis_angle_quat(_Z_AXIS, np.deg2rad(mount_yaw_deg))


def _quat_inv(q: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_negQuat(out, q)
    return out


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, v, q)
    return out


def _flange_target_from_sharpa_root(
    pos_root_world: np.ndarray, quat_root_world: np.ndarray,
    mount_yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Given the desired Sharpa root pose in the dual_ur3e world frame,
    compute the corresponding attachment_site (flange) target pose.

    flange = root · inv(flange_to_root_offset)
        offset is Translate(0,0,HAND_OFFSET) then RotZ(mount_yaw_deg) in
        the flange frame (135° right, 45° left).
    """
    q_off_inv = _quat_inv(_flange_offset_quat(mount_yaw_deg))
    quat_flange = _quat_mul(quat_root_world, q_off_inv)
    # offset_pos_in_flange = (0, 0, HAND_OFFSET); world offset = R_flange · that
    offset_world = _quat_rotate(quat_flange, np.array([0.0, 0.0, HAND_OFFSET]))
    pos_flange = pos_root_world - offset_world
    return pos_flange, quat_flange


# ---------------------------------------------------------------------------
# Collision-aware IK (mink)
# ---------------------------------------------------------------------------

MINK_SOLVER = "daqp"  # the only qpsolvers backend installed in the mujoco env


def _collision_geom_ids(
    model: mujoco.MjModel, include_prefixes, exclude_prefixes=(),
) -> list[int]:
    """Collision-geom ids whose owning body name starts with any of
    ``include_prefixes`` and none of ``exclude_prefixes``.

    A geom counts as a collision geom iff (contype | conaffinity) != 0 — this
    catches both the UR3e arm capsules (group 3) and the static environment
    boxes (group 0, contype=conaffinity=1) while skipping the visual-only
    meshes (contype=conaffinity=0, groups 1/2). Geoms are returned by integer
    id because the UR3e collision geoms are unnamed.
    """
    ids = []
    for g in range(model.ngeom):
        if (int(model.geom_contype[g]) | int(model.geom_conaffinity[g])) == 0:
            continue
        bname = model.body(model.geom_bodyid[g]).name
        if not any(bname.startswith(p) for p in include_prefixes):
            continue
        if any(bname.startswith(p) for p in exclude_prefixes):
            continue
        ids.append(g)
    return ids


def _collision_groups(model: mujoco.MjModel, side: str) -> dict:
    """Group collision-geom ids for the active ``side`` into
    active_arm / active_hand / other_arm / other_hand / env.

    The arm group is the "<side>_..." bodies minus the "<side>_hand_..."
    subtree, so the two wrist couplers (which build_scene mounts on
    <side>_wrist_3_link) fall into the arm group. ``env`` is every collision
    geom owned by the world body (floor, table, legs, walls, foam,
    breadboard, supports, both mounting blocks).
    """
    other = "right" if side == "left" else "left"
    return {
        "active_arm": _collision_geom_ids(model, (f"{side}_",), (f"{side}_hand_",)),
        "active_hand": _collision_geom_ids(model, (f"{side}_hand_",)),
        "other_arm": _collision_geom_ids(model, (f"{other}_",), (f"{other}_hand_",)),
        "other_hand": _collision_geom_ids(model, (f"{other}_hand_",)),
        "env": [
            g for g in range(model.ngeom)
            if model.body(model.geom_bodyid[g]).name == "world"
            and (int(model.geom_contype[g]) | int(model.geom_conaffinity[g])) != 0
        ],
    }


class MinkArmIK:
    """mink differential-IK for one UR3e arm (6 DOF) with self/environment
    collision avoidance.

    The full dual-arm model is wrapped in a single ``mink.Configuration``. A
    ``DofFreezingTask`` (enforced as a hard equality constraint) pins every dof
    except the active arm's 6 joints, so the QP only ever moves that arm; the
    finger angles and the idle arm are written straight into the configuration
    each frame and held there. A ``FrameTask`` drives the
    ``<side>_attachment_site`` to the requested world pose, a low-cost
    ``PostureTask`` regularises the redundant nullspace, and a (rebuildable)
    ``CollisionAvoidanceLimit`` stops the arm from intersecting itself, the
    table/blocks and — if enabled — the other arm.

    The solve is an iterated Gauss-Newton step: each iteration moves
    Δq ≈ -gain · pose_error, so the FrameTask ``gain`` (≈0.5) is the step
    fraction — a full step (gain=1) overshoots and oscillates near the reach
    boundary. When a target is unreachable (out of range, or held off by a
    collision constraint) the loop stops once the step shrinks below
    ``step_tol`` and reports the residual rather than grinding to max_iters.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        side: str,
        arm_qadr: np.ndarray,
        finger_qadr: np.ndarray,
        site_id: int,
        home_qpos: np.ndarray,
        *,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        posture_cost: float = 1e-3,
        frame_gain: float = 0.5,
        lm_damping: float = 1e-3,
        config_limit_gain: float = 0.95,
        arm_max_vel: float = 1.0,
    ) -> None:
        self.model = model
        self.side = side
        self.arm_qadr = np.asarray(arm_qadr, dtype=np.int32)
        self.finger_qadr = np.asarray(finger_qadr, dtype=np.int32)
        self.site_id = int(site_id)
        self.home_qpos = np.asarray(home_qpos, dtype=np.float64).copy()
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, self.site_id)
        # Names of the six arm joints (whose qpos addresses are arm_qadr), used
        # to build the per-iteration velocity cap below.
        arm_joint_names = []
        for a in self.arm_qadr:
            j = int(np.where(model.jnt_qposadr == a)[0][0])
            arm_joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))

        # Hinge joints only: jnt_qposadr == jnt_dofadr for this model, so the
        # arm's qpos addresses double as its velocity (dof) indices.
        self.arm_dof = [int(d) for d in self.arm_qadr]

        # mink's ConfigurationLimit AND Configuration.check_limits consider
        # every *limited* joint. But the retargeted finger angles routinely
        # exceed the Sharpa hand's MJCF ranges, and since every non-arm dof is
        # hard-frozen the limit would demand a corrective Δq the freeze forbids
        # → an infeasible QP (and check_limits, run every solve_ik, would log a
        # debug line per out-of-range finger per iteration). Drop the joint
        # limits on all non-arm joints (they're pinned anyway) BEFORE building
        # the Configuration so its cached limited-joint snapshot also excludes
        # them; ConfigurationLimit (built below) then only ever governs the 6
        # active arm joints, which keep their real ranges.
        arm_dof_set = set(self.arm_dof)
        for j in range(model.njnt):
            if int(model.jnt_dofadr[j]) not in arm_dof_set:
                model.jnt_limited[j] = 0

        self.configuration = mink.Configuration(model, q=self.home_qpos)
        self.frame_task = mink.FrameTask(
            frame_name=site_name, frame_type="site",
            position_cost=position_cost, orientation_cost=orientation_cost,
            gain=frame_gain, lm_damping=lm_damping,
        )
        frozen = [d for d in range(model.nv) if d not in arm_dof_set]
        self.frozen_dof = np.asarray(frozen, dtype=np.int32)
        self.freeze_task = mink.DofFreezingTask(model=model, dof_indices=frozen)
        self.posture_task = mink.PostureTask(model, cost=posture_cost)
        self.posture_task.set_target(self.home_qpos)

        self.config_limit = mink.ConfigurationLimit(model, gain=config_limit_gain)
        # Per-iteration step cap on the arm joints. Collision avoidance is a
        # velocity-level (linearized) constraint: without bounded steps a big
        # Gauss-Newton jump leaps straight through a contact before the
        # constraint can react. The cap keeps each step small enough that the
        # linearization stays valid, so the hard collision constraints actually
        # hold. Only applied when collision avoidance is on (see _limits).
        self.velocity_limit = mink.VelocityLimit(
            model, {name: arm_max_vel for name in arm_joint_names}
        )
        self._groups = _collision_groups(model, side)
        self.collision_limit = None
        self.collision_pairs_n = 0
        # Count of QP iterations where daqp failed and we fell back to the
        # collision-free step, and the worst (smallest) signed clearance seen
        # across a solve_all_frames pass. Both reset by reset().
        self.qp_failures = 0
        self.worst_clearance = float("inf")

    def configure_collision(
        self,
        *,
        enable: bool = True,
        self_collision: bool = True,
        env_collision: bool = True,
        other_arm_collision: bool = False,
        min_distance: float = 0.005,
        detection_distance: float = 0.05,
        gain: float = 0.85,
    ) -> int:
        """(Re)build the CollisionAvoidanceLimit; return the number of geom
        pairs it tracks (0 when disabled). Cheap enough to call live from the
        GUI between solves.

        ``self_collision`` adds active-arm-vs-active-arm (mink auto-drops the
        adjacent/welded link pairs); ``env_collision`` adds active arm vs the
        static world; ``other_arm_collision`` adds active arm vs the parked
        other arm (off by default — the other arm sits far from the workspace).

        Only the arm's collision *primitives* (the UR3e link capsules, the
        wrist-3 cylinder and the two couplers) are used — never the Sharpa
        hand's detailed collision *meshes*. mujoco.mj_geomDistance silently
        returns 0.0 for those meshes against the large table box at the query
        margins mink uses, which would make the solver think the hand is
        permanently in contact with the world. The wrist-3 cylinder + couplers
        already guard the hand base; the frozen fingers are downstream.
        """
        g = self._groups
        arm = g["active_arm"]
        pairs: list = []
        if enable and self_collision and arm:
            pairs.append((arm, arm))
        if enable and env_collision and g["env"] and arm:
            pairs.append((arm, g["env"]))
        if enable and other_arm_collision and g["other_arm"] and arm:
            pairs.append((arm, g["other_arm"]))
        if not pairs:
            self.collision_limit = None
            self.collision_pairs_n = 0
            return 0
        self.collision_limit = mink.CollisionAvoidanceLimit(
            self.model, pairs, gain=gain,
            minimum_distance_from_collisions=min_distance,
            collision_detection_distance=detection_distance,
        )
        self.collision_pairs_n = int(self.collision_limit.max_num_contacts)
        return self.collision_pairs_n

    def _limits(self) -> list:
        lims = [self.config_limit]
        if self.collision_limit is not None:
            # The velocity cap is what makes the linearized collision
            # constraints reliable, so pair the two.
            lims.append(self.velocity_limit)
            lims.append(self.collision_limit)
        return lims

    def reset(self, arm_seed: np.ndarray | None = None) -> None:
        """Park every dof at the home keyframe; optionally seed the active
        arm at ``arm_seed`` (the IK warmstart for the first solved frame)."""
        q = self.home_qpos.copy()
        if arm_seed is not None:
            q[self.arm_qadr] = arm_seed
        self.configuration.update(q=q)
        self.qp_failures = 0
        self.worst_clearance = float("inf")

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        finger_qpos: np.ndarray,
        *,
        arm_seed: np.ndarray | None = None,
        max_iters: int = 120,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
        step_tol: float = 1e-6,
        dt: float = 0.1,
    ) -> tuple[np.ndarray, float, float, int]:
        """Solve IK for one frame.

        ``finger_qpos`` (22,) is written into the configuration and frozen;
        ``arm_seed`` (6,) optionally re-seeds the active arm (pass it on the
        first solved frame, then None to warmstart from the previous frame's
        solution).

        ``dt`` cancels out of the task step and the collision *position* bound,
        but it is NOT a free knob once collision avoidance is on: the
        VelocityLimit caps each step at ``arm_max_vel · dt`` rad and the
        CollisionAvoidanceLimit's per-step slack scales as ``1/dt``. The
        default 0.1 (→ 0.1 rad/iter cap at arm_max_vel=1.0) is the tested
        operating point; change it together with --ik-max-vel.

        ``pos_err`` is the true Euclidean site-position error (m) and
        ``rot_err`` the geodesic orientation error (rad). Returns
        (arm_qpos(6,), pos_err, rot_err, iters).
        """
        data = self.configuration.data
        if arm_seed is not None:
            data.qpos[self.arm_qadr] = arm_seed
        data.qpos[self.finger_qadr] = finger_qpos
        self.configuration.update()

        rot = mink.SO3(wxyz=np.asarray(target_quat, dtype=np.float64)).normalize()
        target = mink.SE3.from_rotation_and_translation(
            rotation=rot, translation=np.asarray(target_pos, dtype=np.float64),
        )
        self.frame_task.set_target(target)
        target_pos_arr = np.asarray(target_pos, dtype=np.float64)

        limits = self._limits()
        # Collision-free limit set, used to recover from a daqp infeasibility
        # (collision + joint-range + freeze constraints can conflict right at a
        # contact boundary). The dropped step keeps the velocity cap so it
        # stays bounded; the collision limit re-engages on the next iteration.
        fallback = [self.config_limit]
        if self.collision_limit is not None:
            fallback.append(self.velocity_limit)
        tasks = [self.frame_task, self.posture_task]
        pos_err = rot_err = 0.0
        it = 0
        for it in range(max_iters):
            err = self.frame_task.compute_error(self.configuration)
            # True Cartesian site-position error (not the SE3 twist translation
            # component, which differs from Euclidean distance when a rotation
            # residual coexists). rot_err is the geodesic angle (twist rotation
            # part).
            pos_err = float(np.linalg.norm(
                target_pos_arr - self.configuration.data.site(self.site_id).xpos
            ))
            rot_err = float(np.linalg.norm(err[3:]))
            if pos_err < tol_pos and rot_err < tol_rot:
                break
            try:
                vel = mink.solve_ik(
                    self.configuration, tasks, dt, MINK_SOLVER,
                    limits=limits, constraints=[self.freeze_task],
                )
            except mink.exceptions.NoSolutionFound:
                self.qp_failures += 1
                try:
                    vel = mink.solve_ik(
                        self.configuration, tasks, dt, MINK_SOLVER,
                        limits=fallback, constraints=[self.freeze_task],
                    )
                except mink.exceptions.NoSolutionFound:
                    break  # give up on this frame; keep the last feasible pose
            # DofFreezingTask drives these to ~0 already; zero them to keep
            # float noise from accumulating on the idle dofs.
            vel[self.frozen_dof] = 0.0
            self.configuration.integrate_inplace(vel, dt)
            if float(np.linalg.norm(vel[self.arm_dof])) * dt < step_tol:
                # No further progress (converged, joint-range-limited, or held
                # off by a collision constraint): re-read residual and stop.
                err = self.frame_task.compute_error(self.configuration)
                pos_err = float(np.linalg.norm(
                    target_pos_arr - self.configuration.data.site(self.site_id).xpos
                ))
                rot_err = float(np.linalg.norm(err[3:]))
                break
        return data.qpos[self.arm_qadr].copy(), pos_err, rot_err, it

    def achieved_wrist_world(self) -> np.ndarray:
        """World position of the sharpa root for the current configuration:
        site_pos + R_site · [0, 0, HAND_OFFSET]."""
        site = self.configuration.data.site(self.site_id)
        mat = np.asarray(site.xmat).reshape(3, 3)
        return np.asarray(site.xpos) + mat @ np.array([0.0, 0.0, HAND_OFFSET])

    def min_clearance(self, distmax: float = 0.2) -> float:
        """Smallest signed distance over all configured collision pairs for the
        current configuration (negative = penetration). Diagnostic only; NaN
        when collision avoidance is disabled. ``distmax`` is kept modest
        because mj_geomDistance loses accuracy for large query margins; pairs
        farther apart than this clamp to ``distmax`` (they aren't the min)."""
        if self.collision_limit is None:
            return float("nan")
        data = self.configuration.data
        mujoco.mj_forward(self.model, data)
        fromto = np.zeros(6)
        worst = distmax
        for g1, g2 in self.collision_limit.geom_id_pairs:
            d = mujoco.mj_geomDistance(self.model, data, int(g1), int(g2), distmax, fromto)
            worst = min(worst, d)
        return worst


# ---------------------------------------------------------------------------
# Index resolution
# ---------------------------------------------------------------------------


def _resolve_indices(model: mujoco.MjModel, side: str) -> dict:
    """Look up qposadr/site/body indices used by the replay loop.

    ``side`` is "right" or "left"; it selects the arm joints, the Sharpa
    finger joints, and the attachment_site by name prefix.
    """
    arm_joints = [
        f"{side}_shoulder_pan_joint",
        f"{side}_shoulder_lift_joint",
        f"{side}_elbow_joint",
        f"{side}_wrist_1_joint",
        f"{side}_wrist_2_joint",
        f"{side}_wrist_3_joint",
    ]
    arm_qadr = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
         for n in arm_joints],
        dtype=np.int32,
    )

    # Sharpa joint names carry the side prefix and are attached under a
    # "<side>_hand_" namespace, e.g. right_hand_right_thumb_CMC_FE.
    finger_joints = [
        f"{side}_thumb_CMC_FE", f"{side}_thumb_CMC_AA", f"{side}_thumb_MCP_FE",
        f"{side}_thumb_MCP_AA", f"{side}_thumb_IP",
        f"{side}_index_MCP_FE", f"{side}_index_MCP_AA", f"{side}_index_PIP",
        f"{side}_index_DIP",
        f"{side}_middle_MCP_FE", f"{side}_middle_MCP_AA", f"{side}_middle_PIP",
        f"{side}_middle_DIP",
        f"{side}_ring_MCP_FE", f"{side}_ring_MCP_AA", f"{side}_ring_PIP",
        f"{side}_ring_DIP",
        f"{side}_pinky_CMC", f"{side}_pinky_MCP_FE", f"{side}_pinky_MCP_AA",
        f"{side}_pinky_PIP", f"{side}_pinky_DIP",
    ]
    finger_qadr = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_" + n,
        )] for n in finger_joints],
        dtype=np.int32,
    )

    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_attachment_site"
    )
    return {
        "arm_qadr": arm_qadr, "finger_qadr": finger_qadr, "site_id": site_id,
        "arm_joint_names": arm_joints, "finger_joint_names": finger_joints,
    }


# ---------------------------------------------------------------------------
# Workspace placement
# ---------------------------------------------------------------------------


def _workspace_rot_quat(
    yaw_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0
) -> np.ndarray:
    """World-frame workspace rotation RotZ(yaw) · RotY(pitch) · RotX(roll).

    Returned as a wxyz quaternion. With pitch=roll=0 this reduces to the
    original yaw-only behavior.
    """
    q_yaw = _axis_angle_quat(_Z_AXIS, np.deg2rad(yaw_deg))
    q_pitch = _axis_angle_quat(_Y_AXIS, np.deg2rad(pitch_deg))
    q_roll = _axis_angle_quat(_X_AXIS, np.deg2rad(roll_deg))
    return _quat_mul(q_yaw, _quat_mul(q_pitch, q_roll))


def _workspace_transform(
    spider_pos: np.ndarray,
    spider_quat: np.ndarray,
    *,
    offset: np.ndarray,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply T_workspace = Translate(offset) · RotZ(yaw)·RotY(pitch)·RotX(roll)
    to a spider pose."""
    q_rot = _workspace_rot_quat(yaw_deg, pitch_deg, roll_deg)
    new_pos = _quat_rotate(q_rot, spider_pos) + offset
    new_quat = _quat_mul(q_rot, spider_quat)
    return new_pos, new_quat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_workspace_defaults(args: argparse.Namespace) -> None:
    """Fill in workspace/start_frame defaults from a reference retarget npz
    if --reference was given (or auto-detected next to --traj). Explicit
    CLI overrides (any flag whose value is not None) win unconditionally.
    """
    ref_path: Path | None = args.reference
    auto = False
    if ref_path is None:
        candidate = args.traj.parent / RETARGET_FILENAME
        if candidate.exists() and candidate.resolve() != args.traj.resolve():
            ref_path = candidate
            auto = True

    ref = _load_reference_defaults(ref_path) if ref_path is not None else None
    have_reference = ref is not None
    if have_reference:
        tag = "auto-detected" if auto else "explicit"
        print(
            f"Using reference retarget ({tag}): {ref_path}\n"
            f"  workspace=({ref['x']:.3f}, {ref['y']:.3f}, {ref['z']:.3f})  "
            f"yaw={ref['yaw']:.1f} pitch={ref['pitch']:.1f} roll={ref['roll']:.1f}  "
            f"start_frame={ref['start_frame']}"
        )
    else:
        preset = PRESET_FRONT_OF_ARM[args.side]
        ref = {
            "x": preset["x"], "y": preset["y"], "z": preset["z"],
            "yaw": preset["yaw"], "pitch": preset["pitch"],
            "roll": preset["roll"], "start_frame": DEFAULT_START_FRAME,
        }

    if args.workspace_x is None:
        args.workspace_x = ref["x"]
    if args.workspace_y is None:
        args.workspace_y = ref["y"]
    if args.workspace_z is None:
        args.workspace_z = ref["z"]
    if args.workspace_yaw is None:
        args.workspace_yaw = ref["yaw"]
    if args.workspace_pitch is None:
        args.workspace_pitch = ref["pitch"]
    if args.workspace_roll is None:
        args.workspace_roll = ref["roll"]
    args.start_frame_from_reference = have_reference and args.start_frame is None
    if args.start_frame is None:
        args.start_frame = ref["start_frame"]


def main(args: argparse.Namespace) -> None:
    _resolve_workspace_defaults(args)

    print("Building dual_ur3e scene...")
    spec = build()
    model = spec.compile()

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data = mujoco.MjData(model)
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    else:
        mujoco.mj_resetData(model, data)

    idx = _resolve_indices(model, args.side)
    arm_qadr = idx["arm_qadr"]
    finger_qadr = idx["finger_qadr"]
    site_id = idx["site_id"]
    arm_joint_names = idx["arm_joint_names"]
    finger_joint_names = idx["finger_joint_names"]
    mount_yaw_deg = HAND_MOUNT_YAW_DEG[args.side]
    print(f"Retargeting onto the {args.side} arm + {args.side} Sharpa hand.")

    print(f"Loading retarget trajectory: {args.traj}")
    with np.load(args.traj) as npz:
        qpos = np.asarray(npz["qpos"])
    if qpos.ndim == 3:
        n_stages, steps_per_stage, _ = qpos.shape
        qpos = qpos.reshape(-1, qpos.shape[-1])
    else:
        n_stages, steps_per_stage = 1, qpos.shape[0]
    n_frames = qpos.shape[0]
    print(
        f"Trajectory: {n_stages} stage(s) x {steps_per_stage} step(s) "
        f"= {n_frames} frames"
    )

    sim_dt = 0.005  # spider config.yaml default
    duration = n_frames * sim_dt

    if not (0 <= args.start_frame < n_frames):
        if args.start_frame_from_reference:
            clamped = max(0, min(args.start_frame, n_frames - 1))
            print(
                f"  reference start_frame={args.start_frame} out of range "
                f"[0, {n_frames}); clamping to {clamped}"
            )
            args.start_frame = clamped
        else:
            raise ValueError(
                f"--start-frame={args.start_frame} out of range [0, {n_frames})"
            )

    solved_arm_qpos = np.zeros((n_frames, 6), dtype=np.float64)
    home_arm_qpos = data.qpos[arm_qadr].copy()  # home keyframe values

    # Build the collision-aware mink IK solver. It wraps the full model in one
    # mink.Configuration and only ever moves the active arm.
    ik_solver = MinkArmIK(
        model, args.side, arm_qadr, finger_qadr, site_id, data.qpos.copy(),
        frame_gain=args.ik_gain, lm_damping=args.ik_lm_damping,
        posture_cost=args.posture_cost, arm_max_vel=args.ik_max_vel,
    )
    npair = ik_solver.configure_collision(
        enable=args.collision,
        self_collision=not args.no_self_collision,
        env_collision=not args.no_env_collision,
        other_arm_collision=args.collision_other_arm,
        min_distance=args.collision_min_dist,
        detection_distance=args.collision_detect_dist,
        gain=args.collision_gain,
    )
    if args.collision:
        bits = []
        if not args.no_self_collision:
            bits.append("self")
        if not args.no_env_collision:
            bits.append("env")
        if args.collision_other_arm:
            bits.append("other-arm")
        print(
            f"IK backend: mink — collision avoidance ON "
            f"[{'+'.join(bits) or 'none'}], {npair} geom pairs, "
            f"clearance>={args.collision_min_dist*1000:.0f}mm, "
            f"gain={args.ik_gain}"
        )
    else:
        print(f"IK backend: mink — collision avoidance OFF, gain={args.ik_gain}")

    def solve_all_frames(
        wx: float, wy: float, wz: float, yaw_deg: float,
        pitch_deg: float, roll_deg: float,
        ik_iters: int, start_frame: int,
    ) -> tuple[float, float, float, np.ndarray]:
        """Re-run IK for the trajectory at the given workspace transform.

        ``start_frame`` is the frame whose wrist is anchored at
        ``(wx, wy, wz)``. IK is solved for f >= start_frame; for
        f < start_frame the arm just holds the start_frame pose, so the
        arm starts moving from start_frame onwards. Hand finger angles
        always come from ``qpos[f, 6:28]`` regardless.

        ``yaw_deg``/``pitch_deg``/``roll_deg`` rotate the rest of the
        trajectory around the anchor point (RotZ·RotY·RotX, world frame).

        Returns (start_frame_pos_err, worst_pos_err, worst_rot_err,
        start_frame_wrist_world).
        """
        offset = np.array([wx, wy, wz], dtype=np.float64)
        wrist_anchor_spider = qpos[start_frame, 0:3].copy()

        warmstart = home_arm_qpos.copy()
        worst_pos = 0.0
        worst_rot = 0.0
        start_pos_err = 0.0
        start_wrist = np.zeros(3)
        # mink warmstarts from the persistent configuration: park every idle
        # dof at home before the chain so re-solves (GUI recompute) start clean.
        ik_solver.reset()
        for f in range(start_frame, n_frames):
            spider_pos, spider_quat = _spider_root_pose(qpos[f])
            spider_pos_rel = spider_pos - wrist_anchor_spider
            root_world_pos, root_world_quat = _workspace_transform(
                spider_pos_rel, spider_quat, offset=offset,
                yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg,
            )
            flange_pos, flange_quat = _flange_target_from_sharpa_root(
                root_world_pos, root_world_quat, mount_yaw_deg
            )
            finger_vals = qpos[f, SPIDER_FINGER_SLICE]
            # Seed the arm only on the first solved frame; afterwards the
            # persistent mink configuration warmstarts from the previous
            # solution. Collision/joint limits are baked into the QP. The
            # start frame is a big move from home (mink takes small capped
            # steps), so give it a larger iteration budget than the cheap
            # warmstarted frames.
            seed = warmstart if f == start_frame else None
            f_iters = max(ik_iters, 200) if f == start_frame else ik_iters
            arm_sol, pe, re, _ = ik_solver.solve(
                flange_pos, flange_quat, finger_vals,
                arm_seed=seed, max_iters=f_iters,
            )
            solved_arm_qpos[f] = arm_sol
            if ik_solver.collision_limit is not None:
                ik_solver.worst_clearance = min(
                    ik_solver.worst_clearance, ik_solver.min_clearance()
                )
            # Mirror into the render `data` so downstream readers stay valid.
            data.qpos[arm_qadr] = arm_sol
            data.qpos[finger_qadr] = finger_vals
            warmstart = solved_arm_qpos[f].copy()
            worst_pos = max(worst_pos, pe)
            worst_rot = max(worst_rot, re)
            if f == start_frame:
                start_pos_err = pe
                start_wrist = ik_solver.achieved_wrist_world().copy()
        # Frames before start_frame just hold the start pose.
        if start_frame > 0:
            solved_arm_qpos[:start_frame] = solved_arm_qpos[start_frame]
        return start_pos_err, worst_pos, worst_rot, start_wrist

    # Pre-solve IK for all frames so playback is smooth and IK errors are
    # surfaced once at startup.
    print(
        f"Solving IK for frames {args.start_frame}..{n_frames - 1} "
        f"(arm holds {args.start_frame} for f<{args.start_frame})..."
    )
    _t_solve = time.perf_counter()
    sf_pos, worst_pos, worst_rot, sf_wrist = solve_all_frames(
        args.workspace_x, args.workspace_y, args.workspace_z,
        args.workspace_yaw, args.workspace_pitch, args.workspace_roll,
        args.ik_iters, args.start_frame,
    )
    print(
        f"  solved {n_frames - args.start_frame} frames in "
        f"{time.perf_counter() - _t_solve:.1f}s\n"
        f"  start_frame ({args.start_frame}) pos_err={sf_pos*1000:.2f} mm  "
        f"achieved wrist world=({sf_wrist[0]:.3f}, {sf_wrist[1]:.3f}, {sf_wrist[2]:.3f})\n"
        f"  worst residual across solved frames: "
        f"pos={worst_pos*1000:.2f} mm, rot={np.rad2deg(worst_rot):.2f} deg"
    )
    if ik_solver.collision_limit is not None:
        clr = ik_solver.worst_clearance
        flag = "  (PENETRATION!)" if clr < -1e-4 else "  (collision-free)"
        qpf = (
            f", {ik_solver.qp_failures} QP fallbacks"
            if ik_solver.qp_failures else ""
        )
        print(
            f"  collision: {ik_solver.collision_pairs_n} geom pairs tracked, "
            f"worst signed clearance across all solved frames = "
            f"{clr*1000:.1f} mm{flag}{qpf}"
        )

    if args.solve_only:
        print("--solve-only: IK solved, skipping the viser GUI.")
        return

    # Reset data to home for replay rendering.
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)

    server = viser.ViserServer()
    mj_scene = ViserMujocoScene(server, model, num_envs=1)
    # Default camera-tracking shifts the entire scene so the first dynamic
    # body sits at the viewer origin — that makes our markers (which we
    # place in true world coords) appear in the wrong spot. Disable it so
    # what you see matches the underlying world frame.
    mj_scene.camera_tracking_enabled = False

    # Parent both reference frames under /fixed_bodies so that if tracking
    # ever gets toggled back on (its position is set to the tracking offset
    # each frame), our markers slide along with the rest of the scene.
    # Static world-origin marker (permanent reference at world (0, 0, 0)).
    server.scene.add_frame(
        "/fixed_bodies/world_origin",
        show_axes=True,
        axes_length=0.30,
        axes_radius=0.008,
        position=(0.0, 0.0, 0.0),
    )

    # Live frame: where the wrist (sharpa root) sits at frame 0 after the
    # workspace transform. Position = (ws_x, ws_y, ws_z); RotZ(ws_yaw).
    target_frame = server.scene.add_frame(
        "/fixed_bodies/workspace_target",
        show_axes=True,
        axes_length=0.15,
        axes_radius=0.004,
        position=(args.workspace_x, args.workspace_y, args.workspace_z),
    )

    def update_target_frame() -> None:
        target_frame.position = (
            float(ws_x.value), float(ws_y.value), float(ws_z.value),
        )
        q = _workspace_rot_quat(
            float(ws_yaw.value), float(ws_pitch.value), float(ws_roll.value)
        )
        target_frame.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    frame_idx = [0]
    playing = [True]
    speed = [1.0]
    looping = [True]
    accumulator = [0.0]
    needs_render = [True]

    tabs = mj_scene.create_visualization_gui()
    with tabs.add_tab("Playback", icon=viser.Icon.PLAYER_PLAY):
        timeline = server.gui.add_slider(
            "Frame", min=0, max=n_frames - 1, step=1, initial_value=0,
        )
        time_label = server.gui.add_html("")
        play_btn = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)

        @play_btn.on_click
        def _(_) -> None:
            playing[0] = not playing[0]
            play_btn.label = "Pause" if playing[0] else "Play"
            play_btn.icon = (
                viser.Icon.PLAYER_PAUSE if playing[0] else viser.Icon.PLAYER_PLAY
            )

        speed_btns = server.gui.add_button_group(
            "Speed", options=["0.25x", "0.5x", "1x", "2x", "4x"]
        )

        @speed_btns.on_click
        def _(event) -> None:
            speed[0] = float(event.target.value.replace("x", ""))

        loop_cb = server.gui.add_checkbox("Loop", initial_value=True)

        @loop_cb.on_update
        def _(_) -> None:
            looping[0] = loop_cb.value

        @timeline.on_update
        def _(_) -> None:
            frame_idx[0] = int(timeline.value)
            needs_render[0] = True

    with tabs.add_tab("Workspace", icon=viser.Icon.ADJUSTMENTS):
        ws_x = server.gui.add_slider(
            "x (m)", min=-0.8, max=0.8, step=0.005,
            initial_value=float(args.workspace_x),
        )
        ws_y = server.gui.add_slider(
            "y (m)", min=-0.8, max=0.8, step=0.005,
            initial_value=float(args.workspace_y),
        )
        ws_z = server.gui.add_slider(
            "z (m)", min=0.0, max=1.6, step=0.005,
            initial_value=float(args.workspace_z),
        )
        ws_yaw = server.gui.add_slider(
            "yaw (deg)", min=-180.0, max=180.0, step=1.0,
            initial_value=float(args.workspace_yaw),
        )
        ws_pitch = server.gui.add_slider(
            "pitch (deg)", min=-180.0, max=180.0, step=1.0,
            initial_value=float(args.workspace_pitch),
        )
        ws_roll = server.gui.add_slider(
            "roll (deg)", min=-180.0, max=180.0, step=1.0,
            initial_value=float(args.workspace_roll),
        )
        ws_start_frame = server.gui.add_slider(
            "start frame (arm holds before)",
            min=0, max=n_frames - 1, step=1,
            initial_value=int(args.start_frame),
        )
        ik_iters_slider = server.gui.add_slider(
            "ik iters", min=8, max=256, step=8,
            initial_value=int(args.ik_iters),
        )
        ik_status = server.gui.add_html("")

        # --- Collision avoidance. These rebuild the mink CollisionAvoidanceLimit
        # on the next "Recompute IK". Disable + recompute for a fast workspace
        # tune, then re-enable for the final collision-free solve.
        col_enable = server.gui.add_checkbox(
            "collision avoidance", initial_value=bool(args.collision),
        )
        col_self = server.gui.add_checkbox(
            "· arm self-collision", initial_value=bool(not args.no_self_collision),
        )
        col_env = server.gui.add_checkbox(
            "· vs table / blocks", initial_value=bool(not args.no_env_collision),
        )
        col_other = server.gui.add_checkbox(
            "· vs other arm", initial_value=bool(args.collision_other_arm),
        )
        col_min = server.gui.add_slider(
            "min clearance (m)", min=0.0, max=0.05, step=0.001,
            initial_value=float(args.collision_min_dist),
        )
        col_detect = server.gui.add_slider(
            "detect distance (m)", min=0.01, max=0.15, step=0.005,
            initial_value=float(args.collision_detect_dist),
        )
        col_status = server.gui.add_html(
            f'<span style="font-size:0.85em">{ik_solver.collision_pairs_n} '
            f"geom pairs</span>"
        )

        recompute_btn = server.gui.add_button(
            "Recompute IK", icon=viser.Icon.REFRESH
        )

        default_out = str(args.traj.parent / RETARGET_FILENAME)
        save_path = server.gui.add_text("output path", initial_value=default_out)
        save_status = server.gui.add_html("")
        save_btn = server.gui.add_button(
            "Save retarget", icon=viser.Icon.DEVICE_FLOPPY
        )

        @save_btn.on_click
        def _(_) -> None:
            save_btn.disabled = True
            try:
                out = Path(save_path.value).expanduser()
                out.parent.mkdir(parents=True, exist_ok=True)
                arm_qpos_out = solved_arm_qpos.copy()
                finger_qpos_out = qpos[:, SPIDER_FINGER_SLICE].astype(np.float64)
                np.savez(
                    out,
                    arm_qpos=arm_qpos_out,
                    finger_qpos=finger_qpos_out,
                    arm_joint_names=np.array(arm_joint_names),
                    finger_joint_names=np.array(finger_joint_names),
                    dt=np.float64(sim_dt),
                    start_frame=np.int64(int(ws_start_frame.value)),
                    workspace_xyz=np.array(
                        [float(ws_x.value), float(ws_y.value), float(ws_z.value)]
                    ),
                    workspace_yaw_deg=np.float64(float(ws_yaw.value)),
                    workspace_pitch_deg=np.float64(float(ws_pitch.value)),
                    workspace_roll_deg=np.float64(float(ws_roll.value)),
                    side=str(args.side),
                    source_traj=str(args.traj),
                )
                save_status.content = (
                    f'<span style="font-size:0.85em;color:#3a7">'
                    f"saved {arm_qpos_out.shape[0]} frames "
                    f"({arm_qpos_out.shape[1]} arm + "
                    f"{finger_qpos_out.shape[1]} finger) → {out}"
                    f"</span>"
                )
            except Exception as e:
                save_status.content = (
                    f'<span style="font-size:0.85em;color:#c33">'
                    f"save failed: {type(e).__name__}: {e}"
                    f"</span>"
                )
            finally:
                save_btn.disabled = False

        for s in (ws_x, ws_y, ws_z, ws_yaw, ws_pitch, ws_roll):
            @s.on_update
            def _(_) -> None:
                update_target_frame()

        @recompute_btn.on_click
        def _(_) -> None:
            recompute_btn.disabled = True
            ik_status.content = (
                '<span style="font-size:0.85em;color:#888">Solving…</span>'
            )
            try:
                t0 = time.perf_counter()
                # Rebuild the mink collision limit from the live GUI controls
                # before re-solving.
                npair = ik_solver.configure_collision(
                    enable=bool(col_enable.value),
                    self_collision=bool(col_self.value),
                    env_collision=bool(col_env.value),
                    other_arm_collision=bool(col_other.value),
                    min_distance=float(col_min.value),
                    detection_distance=float(col_detect.value),
                    gain=float(args.collision_gain),
                )
                col_status.content = (
                    f'<span style="font-size:0.85em">{npair} geom pairs '
                    f"{'(on)' if npair else '(off)'}</span>"
                )
                sf_idx = int(ws_start_frame.value)
                sf_pos, w_pos, w_rot, sf_wrist = solve_all_frames(
                    float(ws_x.value), float(ws_y.value), float(ws_z.value),
                    float(ws_yaw.value), float(ws_pitch.value), float(ws_roll.value),
                    int(ik_iters_slider.value), sf_idx,
                )
                dt = time.perf_counter() - t0
                col_line = ""
                if ik_solver.collision_limit is not None:
                    clr = ik_solver.worst_clearance
                    warn = ' style="color:#c33"' if clr < -1e-4 else ""
                    col_line = (
                        f"<br><span{warn}>worst clearance "
                        f"{clr*1000:.1f} mm ({ik_solver.collision_pairs_n} pairs)"
                        f"</span>"
                    )
                ik_status.content = (
                    f'<span style="font-size:0.85em">'
                    f"solved {n_frames - sf_idx} frames in {dt:.2f}s &nbsp;&middot;&nbsp; "
                    f"start ({sf_idx}) err {sf_pos*1000:.1f} mm &nbsp;&middot;&nbsp; "
                    f"worst {w_pos*1000:.1f} mm / {np.rad2deg(w_rot):.1f}°<br>"
                    f"start wrist world: "
                    f"({sf_wrist[0]:.3f}, {sf_wrist[1]:.3f}, {sf_wrist[2]:.3f})"
                    f"{col_line}"
                    f"</span>"
                )
                # Snap timeline to the start frame and pause so the user
                # sees the held pose right away.
                frame_idx[0] = sf_idx
                timeline.value = sf_idx
                accumulator[0] = 0.0
                if playing[0]:
                    playing[0] = False
                    play_btn.label = "Play"
                    play_btn.icon = viser.Icon.PLAYER_PLAY
                needs_render[0] = True
            finally:
                recompute_btn.disabled = False

    def render_frame(f: int) -> None:
        data.qpos[arm_qadr] = solved_arm_qpos[f]
        data.qpos[finger_qadr] = qpos[f, SPIDER_FINGER_SLICE]
        mujoco.mj_forward(model, data)
        mj_scene.update_from_mjdata(data)
        stage = f // steps_per_stage if steps_per_stage > 0 else 0
        step_in_stage = f % steps_per_stage if steps_per_stage > 0 else f
        t = f * sim_dt
        # Per-joint angles in degrees so the user can read the actual values
        # off the timeline.
        q_deg = np.rad2deg(solved_arm_qpos[f])
        joints_html = " &nbsp; ".join(
            f"j{j}={q_deg[j]:+6.1f}°" for j in range(6)
        )
        time_label.content = (
            f'<span style="font-size:0.85em">'
            f"{t:.2f}s / {duration:.2f}s &nbsp;&middot;&nbsp; "
            f"frame {f}/{n_frames - 1} &nbsp;&middot;&nbsp; "
            f"stage {stage}/{max(n_stages - 1, 0)} (step {step_in_stage})"
            f"<br><tt>{joints_html}</tt>"
            f"</span>"
        )

    render_frame(0)

    last_time = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            wall_dt = now - last_time
            last_time = now
            if playing[0]:
                accumulator[0] += wall_dt * speed[0]
                frames_to_advance = int(accumulator[0] / sim_dt)
                if frames_to_advance > 0:
                    accumulator[0] -= frames_to_advance * sim_dt
                    new_idx = frame_idx[0] + frames_to_advance
                    if new_idx >= n_frames:
                        if looping[0]:
                            new_idx = new_idx % n_frames
                        else:
                            new_idx = n_frames - 1
                            playing[0] = False
                            play_btn.label = "Play"
                            play_btn.icon = viser.Icon.PLAYER_PLAY
                    frame_idx[0] = new_idx
                    timeline.value = new_idx
                    render_frame(new_idx)
            elif needs_render[0]:
                render_frame(frame_idx[0])
                needs_render[0] = False
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("\nStopped.")
        server.stop()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    p.add_argument(
        "--side", choices=("right", "left"), default="right",
        help="Which arm + Sharpa hand to retarget onto (default: right).",
    )
    p.add_argument(
        "--reference", type=Path, default=None,
        help=(
            "Path to a previously-saved trajectory_dual_ur3e.npz. "
            "Its workspace_xyz / workspace_yaw_deg / start_frame become the "
            "defaults for --workspace-x/-y/-z, --workspace-yaw, --start-frame "
            "(any flag passed explicitly still wins). When omitted, the script "
            "auto-detects a sibling trajectory_dual_ur3e.npz next to --traj."
        ),
    )
    # Defaults are resolved in _resolve_workspace_defaults: reference file
    # (explicit or auto-detected) → PRESET_FRONT_OF_ARM / DEFAULT_START_FRAME.
    p.add_argument(
        "--workspace-x", type=float, default=None,
        help="dual_ur3e world x (m) of the wrist at frame 0",
    )
    p.add_argument(
        "--workspace-y", type=float, default=None,
        help="dual_ur3e world y (m) of the wrist at frame 0",
    )
    p.add_argument(
        "--workspace-z", type=float, default=None,
        help="dual_ur3e world z (m) of the wrist at frame 0",
    )
    p.add_argument(
        "--workspace-yaw", type=float, default=None,
        help="yaw (deg) rotating the rest of the trajectory around the wrist anchor",
    )
    p.add_argument(
        "--workspace-pitch", type=float, default=None,
        help="pitch (deg) rotating the rest of the trajectory around the wrist anchor",
    )
    p.add_argument(
        "--workspace-roll", type=float, default=None,
        help="roll (deg) rotating the rest of the trajectory around the wrist anchor",
    )
    p.add_argument(
        "--start-frame", type=int, default=None,
        help=(
            "Frame whose wrist is anchored at (workspace-x, -y, -z). "
            "The arm holds this pose for f<start_frame and starts moving "
            "from start_frame onward. Hand fingers play from frame 0 normally."
        ),
    )
    p.add_argument(
        "--ik-iters", type=int, default=64,
        help=(
            "Max IK iterations per frame (early-exits on convergence; default "
            "64). Warmstarted frames converge in a handful. The first solved "
            "frame is a big move from home, so it gets its own larger budget "
            "(max(--ik-iters, 200))."
        ),
    )
    p.add_argument(
        "--ik-gain", type=float, default=0.5,
        help=(
            "mink FrameTask gain = per-iteration step fraction "
            "(Δq ≈ -gain·pose_error). ~0.5 converges without overshoot; "
            ">~0.7 oscillates near the reach boundary."
        ),
    )
    p.add_argument("--ik-lm-damping", type=float, default=1e-3,
                   help="mink FrameTask Levenberg-Marquardt damping (singularity robustness).")
    p.add_argument("--posture-cost", type=float, default=1e-3,
                   help="mink PostureTask cost — weak nullspace bias toward the seed posture.")
    p.add_argument(
        "--ik-max-vel", type=float, default=1.0,
        help=(
            "Per-iteration arm joint step cap (the mink VelocityLimit, applied "
            "only with collision avoidance). Smaller steps keep the linearized "
            "collision constraints accurate — too large and the arm leaps "
            "through contacts before they engage. Default 1.0."
        ),
    )
    # --- Collision avoidance ---
    p.add_argument(
        "--collision", action=argparse.BooleanOptionalAction, default=True,
        help="Enable mink collision avoidance (default on; --no-collision to disable).",
    )
    p.add_argument(
        "--no-self-collision", action="store_true",
        help="Drop arm-vs-arm self-collision constraints (kept on by default).",
    )
    p.add_argument(
        "--no-env-collision", action="store_true",
        help="Drop arm/hand-vs-table/blocks constraints (kept on by default).",
    )
    p.add_argument(
        "--collision-other-arm", action="store_true",
        help=(
            "Also avoid the parked other arm+hand (off by default — it roughly "
            "doubles the geom-pair count and the other arm sits far from the "
            "workspace)."
        ),
    )
    p.add_argument(
        "--collision-min-dist", type=float, default=0.005,
        help="Clearance (m) to maintain between collision geoms (default 0.005).",
    )
    p.add_argument(
        "--collision-detect-dist", type=float, default=0.05,
        help=(
            "Distance (m) within which a geom pair becomes an active constraint "
            "(default 0.05). Larger = smoother avoidance, slower solve."
        ),
    )
    p.add_argument(
        "--collision-gain", type=float, default=0.85,
        help="mink CollisionAvoidanceLimit gain in (0,1] — approach speed toward the bound.",
    )
    p.add_argument(
        "--solve-only", action="store_true",
        help="Solve IK, print residual/collision stats, and exit without the viser GUI.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
