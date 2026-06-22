# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    Browser-based replay evaluation for headless / remote servers.

    Like geort/mocap/replay_evaluation.py, but renders in the browser via viser instead
    of the native SAPIEN viewer -- so it needs no display. It steps through a recorded
    human dataset, runs the trained GeoRT model to retarget each frame to robot qpos,
    and drives the hand URDF. The human target fingertips are overlaid as red points so
    you can visually check how well the robot tips track them.

    Usage (in the `geort` env, on the server):
        python geort/mocap/replay_viser.py -hand wuji -ckpt_tag wuji_mano -data mano_right

    Then forward the port and open the printed URL:
        ssh -L 8080:localhost:8080 <user>@<server>   # browse to http://localhost:8080

    GUI: Play/pause, speed (fps), a frame scrubber, and a toggle for the human markers.
'''

import argparse
import time

import numpy as np
import viser
from viser.extras import ViserUrdf

from geort import load_model, get_config
from geort.utils.path import get_human_data
from geort.env.hand_viser import resolve_urdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hand', type=str, default='wuji')
    parser.add_argument('-ckpt_tag', type=str, default='wuji_mano')
    parser.add_argument('-data', type=str, default='mano_right')
    parser.add_argument('--epoch', type=int, default=0, help='checkpoint epoch (<=0 -> last.pth)')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--fps', type=float, default=20.0)
    args = parser.parse_args()

    config = get_config(args.hand)
    model = load_model(args.ckpt_tag, epoch=args.epoch)

    human_points = np.load(get_human_data(args.data))  # [T, 21, 3]
    T = len(human_points)
    human_ids = [f['human_hand_id'] for f in config['fingertip_link']]
    print(f"Loaded {T} frames from '{args.data}'; retargeting to '{args.hand}'.")

    urdf_path = resolve_urdf_path(config['urdf_path'])
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction('+z')
    urdf = ViserUrdf(server, urdf_or_path=urdf_path)

    # model.forward returns qpos in config['joint_order']; viser wants the URDF's
    # actuated-joint order. Build the permutation between them.
    urdf_joint_names = urdf.get_actuated_joint_names()
    cfg_index = {name: i for i, name in enumerate(config['joint_order'])}
    missing = [n for n in urdf_joint_names if n not in cfg_index]
    if missing:
        raise ValueError(f"URDF actuated joints not in config joint_order: {missing}")
    perm = np.array([cfg_index[n] for n in urdf_joint_names])

    limits = urdf.get_actuated_joint_limits()
    lo = np.array([(limits[n][0] if limits[n][0] is not None else -np.pi) for n in urdf_joint_names])
    hi = np.array([(limits[n][1] if limits[n][1] is not None else np.pi) for n in urdf_joint_names])

    # GUI
    play = server.gui.add_checkbox('Play', initial_value=True)
    fps = server.gui.add_slider('Speed (fps)', min=1.0, max=60.0, step=1.0, initial_value=args.fps)
    frame = server.gui.add_slider('Frame', min=0, max=T - 1, step=1, initial_value=0)
    show_target = server.gui.add_checkbox('Show human target tips', initial_value=True)

    def render_frame(t):
        t = int(t) % T
        kp = human_points[t]                       # [21,3]
        qpos = model.forward(kp)                    # [n_dof] in config joint_order
        qpos_urdf = np.clip(np.asarray(qpos)[perm], lo, hi)
        urdf.update_cfg(qpos_urdf)
        if show_target.value:
            tips = kp[human_ids].astype(np.float32)  # palm frame ~ robot base frame
            server.scene.add_point_cloud(
                '/human_target', points=tips,
                colors=np.tile(np.array([[255, 40, 40]], np.uint8), (len(tips), 1)),
                point_size=0.006,
            )
        else:
            server.scene.add_point_cloud('/human_target', points=np.zeros((0, 3), np.float32),
                                         colors=np.zeros((0, 3), np.uint8), point_size=0.006)

    # Scrubbing while paused updates immediately.
    @frame.on_update
    def _(_):
        if not play.value:
            render_frame(frame.value)

    render_frame(0)
    print(f"\nViser running on port {args.port}. Forward it over SSH and open the URL above.\n")

    t = 0
    while True:
        if play.value:
            render_frame(t)
            frame.value = t           # reflect playback position in the scrubber
            t = (t + 1) % T
        time.sleep(1.0 / max(fps.value, 1.0))


if __name__ == '__main__':
    main()
