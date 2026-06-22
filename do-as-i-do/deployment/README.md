# Deployment

Run a retargeted demo on the real robot. Two stages:

1. **`mujoco_replay/`** — load a spider `trajectory_mjwp.npz` (from the
   `retargeting/` stage), place + IK-solve it onto the dual-UR3e scene with
   collision-aware [`mink`](https://github.com/kevinzakka/mink) IK, preview/tune
   in a viser GUI, and save `trajectory_dual_ur3e.npz` (arm + finger joint
   trajectories).
2. **`robot_replay/`** — stream `trajectory_dual_ur3e.npz` to a real UR3e arm +
   Sharpa Wave hand at 50 Hz.

## Setup

```bash
conda env create -f env/deployment.yml   # see env/README.md
conda activate deployment
```

The Sharpa Wave hand SDK is proprietary and not shipped — drop it into
`robot_replay/Sharpa/`, then copy `robot_replay/config.example.yaml` to
`robot_replay/config.yaml` and fill in your robot IPs / hand serials.

## 1. mujoco_replay (sim → trajectory)

```bash
cd mujoco_replay
python replay_retarget.py --side left --traj /path/to/trajectory_mjwp.npz
```

Opens a viser web GUI at the printed URL. Tune the workspace placement
(x/y/z + yaw/pitch/roll), the start frame, and collision avoidance; hit
**Recompute IK**, then **Save retarget** to write `trajectory_dual_ur3e.npz`
next to the input trajectory.

When tuning the workspace placement, adjust x/y/z and yaw. Only tune pitch and
roll if the gravity alignment (GeoCalib) from reconstruction was significantly
incorrect — otherwise leave them at zero.

Headless solve (no GUI) to check IK residuals / collision clearance:

```bash
python replay_retarget.py --side left --traj .../trajectory_mjwp.npz --solve-only
```

## 2. robot_replay (trajectory → hardware)

```bash
cd robot_replay
cp config.example.yaml config.yaml        # then edit arm_ip / hand_sn

python run_npz.py --side left trajectory_dual_ur3e.npz             # arm only, real-time
python run_npz.py --side left trajectory_dual_ur3e.npz --both      # arm + hand
python run_npz.py --side left trajectory_dual_ur3e.npz --speed 0.5 # half speed
python run_npz.py --side left trajectory_dual_ur3e.npz --dry-run   # validate, no hardware
```

The script connects, homes, moves to the trajectory start, then waits for Enter
before streaming. Home a robot independently with:

```bash
python home.py --side left            # arm + hand
python home.py --side right --arm-only
```

## Safety

`run_npz.py` and `home.py` command real hardware. Verify the home pose, keep an
e-stop in reach, and start with `--dry-run` then a low `--speed`.
