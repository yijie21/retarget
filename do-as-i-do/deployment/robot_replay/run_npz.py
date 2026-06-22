"""Replay a dual_ur3e trajectory on one real arm + hand (left or right).

Merges the former run_left_npz.py / run_right_npz.py — pick the side with
--side {left,right}. Streams at a fixed 50 Hz; --speed resamples the trajectory
in time (not the streaming rate). --arm/--hand/--both selects which subsystem(s)
to drive. --dry-run loads + resamples without connecting to any hardware.

Defaults: --speed 1.0  (real-time)
          --arm        (arm only)

Quick start (arm only, real-time):
    python run_npz.py --side left trajectory_dual_ur3e.npz

More examples:
    python run_npz.py --side right traj.npz --speed 0.5          # arm, half speed
    python run_npz.py --side left  traj.npz --speed 0.25 --both  # arm + hand, quarter speed
    python run_npz.py --side left  traj.npz --hand               # hand only, real-time
    python run_npz.py --side left  traj.npz --dry-run            # validate, no hardware

The script connects, homes, moves to the trajectory start, then waits for you
to press Enter before streaming. Per-side robot IPs / hand serials / servoJ
tuning / home pose come from config.yaml (see config.example.yaml).

The npz must contain arm_qpos (N, 6), finger_qpos (N, 22), and dt (scalar, s) —
exactly what mujoco_replay/replay_retarget.py saves.
"""
import argparse
import time

import numpy as np

from executor import Controller, load_config, ARM_DIM, HAND_DIM, CMD_DIM


DEFAULT_NPZ = "trajectory_dual_ur3e.npz"

STREAMING_HZ = 50
DT = 1.0 / STREAMING_HZ

HOME_SETTLE_S = 3.0
START_SETTLE_S = 2.0

# moveJ speed/acceleration (rad/s, rad/s^2) for positioning moves: homing and
# moving to the trajectory start. ur_rtde defaults are 1.05/1.4; lower = slower/safer.
MOVE_SPEED = 0.5
MOVE_ACCEL = 0.5

# Pause streaming for human confirmation after this NATIVE trajectory frame
# (raw npz row index, before resampling; None to disable). The streamed frame at
# which playback reaches this native frame is computed from --speed. The robot
# holds its last position (servoJ target) while waiting.
PAUSE_AFTER_FRAME = 600


class FrequencyTimer:
    def __init__(self, rate_hz):
        self.period_ns = int(1e9 / rate_hz)

    def start_loop(self):
        self.start_ns = time.time_ns()

    def end_loop(self):
        deadline = self.start_ns + self.period_ns
        while time.time_ns() < deadline:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--side', choices=('left', 'right'), required=True,
                        help='Which arm + hand to drive.')
    parser.add_argument('npz', nargs='?', default=DEFAULT_NPZ,
                        help=f'Path to trajectory npz (default: {DEFAULT_NPZ}).')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (1.0 = real-time, 0.5 = half speed). Default 1.0.')
    parser.add_argument('--config', default=None,
                        help='Path to config.yaml (default: ./config.yaml, else config.example.yaml).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Load + resample the trajectory and print stats; do NOT connect to hardware.')

    target = parser.add_mutually_exclusive_group()
    target.add_argument('--arm', dest='target', action='store_const', const='arm',
                        help='Run arm only (default).')
    target.add_argument('--hand', dest='target', action='store_const', const='hand',
                        help='Run hand only.')
    target.add_argument('--both', dest='target', action='store_const', const='both',
                        help='Run arm and hand.')
    parser.set_defaults(target='arm')
    return parser.parse_args()


def load_and_resample(npz_path, speed):
    data = np.load(npz_path)
    arm = data['arm_qpos']
    hand = data['finger_qpos']
    native_dt = float(data['dt'])

    if arm.shape[0] != hand.shape[0]:
        raise ValueError(f"arm_qpos ({arm.shape}) and finger_qpos ({hand.shape}) length mismatch")
    if arm.shape[1] != ARM_DIM or hand.shape[1] != HAND_DIM:
        raise ValueError(f"expected arm (N,{ARM_DIM}) hand (N,{HAND_DIM}); got {arm.shape}, {hand.shape}")

    traj = np.concatenate([arm, hand], axis=1)
    assert traj.shape[1] == CMD_DIM

    n_native = traj.shape[0]
    duration_native = n_native * native_dt
    duration_play = duration_native / speed
    n_out = max(2, int(round(duration_play * STREAMING_HZ)))

    t_native = np.arange(n_native) * native_dt
    t_play = np.arange(n_out) / STREAMING_HZ
    t_query = np.minimum(t_play * speed, t_native[-1])

    out = np.empty((n_out, CMD_DIM))
    for j in range(CMD_DIM):
        out[:, j] = np.interp(t_query, t_native, traj[:, j])

    # Map the native pause frame to a streamed-frame index via its time on the
    # native clock. t_query is non-decreasing, so searchsorted finds the first
    # streamed frame that reaches native frame PAUSE_AFTER_FRAME.
    pause_idx = None
    if PAUSE_AFTER_FRAME is not None and PAUSE_AFTER_FRAME < n_native:
        t_pause = PAUSE_AFTER_FRAME * native_dt
        pause_idx = min(int(np.searchsorted(t_query, t_pause)), n_out - 1)

    print(f"Loaded {n_native} native frames (dt={native_dt}s, {duration_native:.2f}s @ {1.0/native_dt:.0f} Hz).")
    print(f"Resampled to {n_out} frames at {STREAMING_HZ} Hz, speed {speed}x "
          f"(playback duration {n_out / STREAMING_HZ:.2f}s).")
    if pause_idx is not None:
        print(f"Will pause after native frame {PAUSE_AFTER_FRAME} (streamed frame {pause_idx + 1}/{n_out}).")
    return out, pause_idx


def main():
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")

    commands, pause_idx = load_and_resample(args.npz, args.speed)

    use_arm = args.target in ('arm', 'both')
    use_hand = args.target in ('hand', 'both')
    print(f"Side: {args.side}  Target: {args.target} (arm={use_arm}, hand={use_hand})")

    if args.dry_run:
        print("--dry-run: trajectory loaded + resampled OK, no hardware connection. Exiting.")
        return

    cfg = load_config(args.config)
    ctrl = Controller(side=args.side, cfg=cfg, dt=DT, use_arm=use_arm, use_hand=use_hand)
    try:
        if use_arm:
            print("Homing arm via moveJ...")
            ctrl.rtde_c.moveJ(ctrl.home_arm_q, MOVE_SPEED, MOVE_ACCEL)
        if use_hand:
            print("Homing hand to zeros...")
            err = ctrl.hand.set_joint_position([0.0] * HAND_DIM, True)
            if err.code != 0:
                print(f"hand set_joint_position error: {err.message}")
        time.sleep(HOME_SETTLE_S)

        first = commands[0]
        if use_arm:
            print("Moving arm to trajectory start via moveJ...")
            ctrl.rtde_c.moveJ(first[:ARM_DIM].tolist(), MOVE_SPEED, MOVE_ACCEL)
        if use_hand:
            print("Moving hand to trajectory start...")
            err = ctrl.hand.set_joint_position(first[ARM_DIM:].tolist(), True)
            if err.code != 0:
                print(f"hand set_joint_position error: {err.message}")
        time.sleep(START_SETTLE_S)

        input(f"\nReady. Press Enter to execute {commands.shape[0]} frames at "
              f"{STREAMING_HZ} Hz ({args.speed}x, ~{commands.shape[0]/STREAMING_HZ:.2f}s, "
              f"side={args.side}, target={args.target})... ")

        print("Streaming...")
        timer = FrequencyTimer(STREAMING_HZ)
        n_frames = len(commands)
        for i, cmd in enumerate(commands):
            timer.start_loop()
            ctrl.execute(cmd.tolist())
            timer.end_loop()
            if i == pause_idx and i < n_frames - 1:
                input(f"\nPaused at native frame {PAUSE_AFTER_FRAME} "
                      f"(streamed frame {i + 1}/{n_frames}). Robot holding position. "
                      f"Press Enter to resume... ")

        print("Done streaming.")
    finally:
        ctrl.close(reset_hand=False)


if __name__ == "__main__":
    main()
