"""Unified UR3e arm + Sharpa Wave hand executor for one side (left or right).

Merges the former left_executor.py / right_executor.py, which were byte-identical
except for per-side constants (arm IP, hand serial, servoJ tuning, home pose).
Those now live in config.yaml (see config.example.yaml); select a side with the
``side`` argument.

Command layout (28-dim, all radians):
    cmd[0:6]   -> UR3e joint angles for servoJ
    cmd[6:28]  -> Sharpa 22 joint angles for set_joint_position
"""
import sys
import time
from pathlib import Path

import yaml

import rtde_control
import rtde_receive

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
EXAMPLE_CONFIG = HERE / "config.example.yaml"
DEFAULT_SHARPA_SDK = HERE / "Sharpa" / "SDK" / "SharpaWaveSDK_4_3_4" / "python"

CMD_DIM = 28
ARM_DIM = 6
HAND_DIM = 22


def load_config(path=None):
    """Load the deployment config.

    With no ``path``, uses config.yaml if present, else falls back to the
    committed config.example.yaml (placeholder IPs/serials — fine for --dry-run
    or inspection, but real values are needed to connect to hardware).
    """
    if path is None:
        path = DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else EXAMPLE_CONFIG
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"config not found: {path}. Copy config.example.yaml to config.yaml "
            f"and fill in your robot IPs / hand serials."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def _import_sharpa(sdk_path):
    """Add the vendored Sharpa SDK python bindings to sys.path and import the
    API. Returns (SharpaWaveManager, ControlMode, ControlSource).

    The SDK is not shipped in the repo; drop it into robot_replay/Sharpa/
    (see config.example.yaml: sharpa_sdk_path).
    """
    sdk_path = str(Path(sdk_path).expanduser())
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    from sharpa import SharpaWaveManager, ControlMode, ControlSource
    return SharpaWaveManager, ControlMode, ControlSource


class Controller:
    """One UR3e arm (ur_rtde) + one Sharpa hand, selected by ``side``."""

    def __init__(self, side, cfg, dt, use_arm=True, use_hand=True):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if not (use_arm or use_hand):
            raise ValueError("at least one of use_arm/use_hand must be True")
        self.side = side
        self.dt = dt
        self.use_arm = use_arm
        self.use_hand = use_hand

        side_cfg = cfg[side]
        self.arm_ip = side_cfg["arm_ip"]
        self.hand_sn = side_cfg["hand_sn"]
        self.servoj_lookahead = float(side_cfg.get("servoj_lookahead", 0.1))
        self.servoj_gain = float(side_cfg.get("servoj_gain", 300))
        self.home_arm_q = list(side_cfg["home_arm_q"])
        self.sharpa_sdk_path = cfg.get("sharpa_sdk_path", str(DEFAULT_SHARPA_SDK))

        if use_arm:
            print(f"Connecting UR3e ({side}) at {self.arm_ip}...")
            self.rtde_c = rtde_control.RTDEControlInterface(self.arm_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.arm_ip)
            self.q0 = self.rtde_r.getActualQ()
            print(f"Starting arm Q (rad): {self.q0}")
        else:
            self.rtde_c = None
            self.rtde_r = None
            self.q0 = None

        if use_hand:
            print(f"Connecting Sharpa hand ({side}) {self.hand_sn}...")
            self._sharpa = _import_sharpa(self.sharpa_sdk_path)
            self.hand = self._connect_hand()
        else:
            self._sharpa = None
            self.hand = None

    def _connect_hand(self):
        SharpaWaveManager, ControlMode, ControlSource = self._sharpa
        manager = SharpaWaveManager.get_instance()
        time.sleep(1.0)
        hand = manager.connect(self.hand_sn)
        if hand is None:
            raise RuntimeError(f"Failed to connect Sharpa hand {self.hand_sn}")

        for fn, val, name in [
            (hand.set_control_mode, ControlMode.POSITION, "control_mode"),
            (hand.set_speed_coeff, 0.3, "speed_coeff"),
            (hand.set_current_coeff, 0.6, "current_coeff"),
            (hand.set_control_source, ControlSource.SDK, "control_source"),
        ]:
            err = fn(val)
            if err.code != 0:
                raise RuntimeError(f"Failed to set {name}: {err.message}")

        hand.start()
        hand.set_joint_position([0.0] * HAND_DIM, True)
        return hand

    def execute(self, cmd):
        if len(cmd) != CMD_DIM:
            raise ValueError(f"cmd must be {CMD_DIM}-dim, got {len(cmd)}")

        arm_q = list(cmd[:ARM_DIM])
        hand_q = list(cmd[ARM_DIM:])

        if self.use_arm:
            self.rtde_c.servoJ(
                arm_q, 0.0, 0.0, self.dt, self.servoj_lookahead, self.servoj_gain
            )

        if self.use_hand:
            err = self.hand.set_joint_position(hand_q, True)
            if err.code != 0:
                print(f"hand set_joint_position error: {err.message}")

    def close(self, reset_hand=True):
        if self.use_arm:
            try:
                self.rtde_c.servoStop()
                self.rtde_c.stopScript()
            except Exception as e:
                print(f"arm stop error: {e}")
        if self.use_hand:
            try:
                if reset_hand:
                    self.hand.set_joint_position([0.0] * HAND_DIM, True)
                    time.sleep(0.5)
                self.hand.stop()
                self._sharpa[0].get_instance().disconnect_all()
            except Exception as e:
                print(f"hand stop error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
