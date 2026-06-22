# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    Browser-based hand visualizer for headless / remote servers.

    Unlike geort/env/hand.py (which opens a native SAPIEN/Vulkan window and needs a
    display), this renders the hand URDF in your web browser using viser. It needs no
    display and no GPU: viser runs a small web server here and you view it locally.

    Usage (on the remote server):
        python geort/env/hand_viser.py --hand allegro_right

    Then forward the port from your laptop and open the printed URL:
        ssh -L 8080:localhost:8080 <user>@<server>
        # browse to http://localhost:8080

    The page gives a slider per joint, plus an "Animate" toggle that drives the hand
    with random targets (the same behavior as hand.py's control loop).
'''

import argparse
import time

import numpy as np
import viser
from viser.extras import ViserUrdf

from geort.utils.config_utils import get_config


def resolve_urdf_path(urdf_path):
    '''Configs store URDF paths relative to the repo root (e.g. "./assets/...").'''
    from pathlib import Path

    p = Path(urdf_path)
    if p.is_absolute() and p.exists():
        return p
    # repo root is two levels up from this file: geort/env/hand_viser.py -> repo/
    repo_root = Path(__file__).resolve().parents[2]
    candidate = (repo_root / p).resolve()
    if candidate.exists():
        return candidate
    # fall back to the path relative to the current working directory.
    return p.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='allegro_right')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Bind address. Use 0.0.0.0 so the SSH-forwarded port reaches it.')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    config = get_config(args.hand)
    urdf_path = resolve_urdf_path(config['urdf_path'])
    print(f"Loading URDF: {urdf_path}")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction('+z')

    urdf = ViserUrdf(server, urdf_or_path=urdf_path)

    # yourdfpy drives the model through its own actuated-joint order; respect it.
    joint_names = urdf.get_actuated_joint_names()
    joint_limits = urdf.get_actuated_joint_limits()  # name -> (lower, upper) or None

    sliders = []
    initial = []
    for name in joint_names:
        lo, hi = joint_limits.get(name, (None, None))
        # Continuous joints report no limit; give them a sensible default range.
        if lo is None or hi is None:
            lo, hi = -np.pi, np.pi
        mid = (lo + hi) / 2.0
        initial.append(mid)
        slider = server.gui.add_slider(
            label=name,
            min=float(lo),
            max=float(hi),
            step=1e-3,
            initial_value=float(mid),
        )
        sliders.append(slider)

    initial = np.array(initial, dtype=np.float64)
    lowers = np.array([s.min for s in sliders])
    uppers = np.array([s.max for s in sliders])

    def push_cfg():
        '''Read all slider values and update the rendered configuration.'''
        urdf.update_cfg(np.array([s.value for s in sliders]))

    for s in sliders:
        s.on_update(lambda _: push_cfg())

    animate = server.gui.add_checkbox('Animate (random targets)', initial_value=False)
    reset = server.gui.add_button('Reset to neutral')

    @reset.on_click
    def _(_):
        for s, v in zip(sliders, initial):
            s.value = float(v)
        push_cfg()

    push_cfg()
    print(f"\nViser is running. Open the URL above in your browser "
          f"(forward port {args.port} over SSH if remote).\n")

    # Animation loop: every ~1s pick fresh random targets, then ease toward them.
    current = initial.copy()
    target = initial.copy()
    dt = 1.0 / 30.0
    steps = 0
    while True:
        if animate.value:
            if steps % 30 == 0:
                target = np.random.uniform(lowers + 1e-6, uppers - 1e-6)
            current += np.clip(target - current, -0.05, 0.05)
            for s, v in zip(sliders, current):
                s.value = float(v)  # triggers push_cfg via on_update
            steps += 1
        else:
            current = np.array([s.value for s in sliders])
            steps = 0
        time.sleep(dt)


if __name__ == '__main__':
    main()
