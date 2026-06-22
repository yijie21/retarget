"""Home one UR3e arm and/or Sharpa hand (merges home_left.py / home_right.py).

    python home.py --side left
    python home.py --side right --arm-only
    python home.py --side left  --hand-only

The arm home pose comes from config.yaml (per side); the hand homes to zeros.
"""
import argparse
import time

from executor import Controller, load_config, ARM_DIM, HAND_DIM


DT = 1.0 / 50
HAND_SETTLE_S = 3.0

# moveJ defaults are speed=1.05 rad/s, acceleration=1.4 rad/s^2; lower = slower homing.
MOVEJ_SPEED = 0.5
MOVEJ_ACCEL = 0.5

HOME_HAND_Q = [0.0] * 22


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--side", choices=("left", "right"), required=True,
                        help="Which arm + hand to home.")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: ./config.yaml, else config.example.yaml).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--arm-only", action="store_true", help="Only home the arm.")
    group.add_argument("--hand-only", action="store_true", help="Only home the hand.")
    args = parser.parse_args()

    home_arm = not args.hand_only
    home_hand = not args.arm_only

    cfg = load_config(args.config)
    with Controller(side=args.side, cfg=cfg, dt=DT, use_arm=home_arm, use_hand=home_hand) as ctrl:
        assert len(ctrl.home_arm_q) == ARM_DIM
        assert len(HOME_HAND_Q) == HAND_DIM

        if home_arm:
            print("Homing arm via moveJ...")
            ctrl.rtde_c.moveJ(ctrl.home_arm_q, MOVEJ_SPEED, MOVEJ_ACCEL)
            print("Arm homed.")

        if home_hand:
            print("Homing hand (interpolation mode)...")
            enable_interpolation_mode = True
            err = ctrl.hand.set_joint_position(HOME_HAND_Q, enable_interpolation_mode)
            if err.code != 0:
                print(f"hand set_joint_position error: {err.message}")
            time.sleep(HAND_SETTLE_S)

        print("Homing complete.")


if __name__ == "__main__":
    main()
