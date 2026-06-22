# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    Real-time hand retargeting from a camera, visualized in the browser (viser).

    Pipeline (all local):
        camera (webcam or RealSense) -> MediaPipe HandLandmarker (21 world landmarks)
        -> palm-frame canonicalize + auto-scale (match training distribution)
        -> GeoRT model.forward -> qpos -> EMA smooth -> drive the Wuji hand in viser.
    Your detected hand (21 keypoints) is shown on the left; the retargeted robot on the right.

    Usage:
        # built-in laptop webcam
        python geort/mocap/live_camera_viser.py -hand wuji -ckpt_tag wuji_mano --camera webcam
        # Intel RealSense
        python geort/mocap/live_camera_viser.py -hand wuji -ckpt_tag wuji_mano --camera realsense
        # no camera (sanity check / offline demo using bundled data)
        python geort/mocap/live_camera_viser.py -hand wuji -ckpt_tag wuji_mano --replay mano_gesture

    Then open http://localhost:8080 (forward the port if remote). Show your RIGHT hand to the
    camera; add --mirror if the robot mirrors you. Quality note: webcam keypoints are noisier
    than a glove, and this checkpoint was trained on MANO data — for the cleanest result,
    collect your own data with mediapipe_mocap.py and retrain (see README).
'''

import argparse
import time

import numpy as np
import viser
from viser.extras import ViserUrdf

from geort import get_config
from geort.utils.path import get_data_root
from geort.env.hand_viser import resolve_urdf_path

OFFSET = 0.22                 # half the lateral gap between the two hands (meters)
MP_MCP = [5, 9, 13, 17]       # index/middle/ring/pinky MCP (MediaPipe 21-landmark layout)
DEFAULT_SCALE = 0.117         # target wrist->middle-tip length (mean of the MANO training set)
SKIN = (224, 172, 144)

# MediaPipe 21-landmark hand topology (wrist, thumb, index, middle, ring, pinky).
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def _n(v):
    return v / (np.linalg.norm(v) + 1e-9)


def skeleton_mesh(kp, joint_r=0.0065, bone_r=0.0042, color=SKIN):
    '''Build a solid "hand model" from 21 keypoints: spheres at joints + cylinders along
    bones, concatenated into one trimesh. This is the camera's detected pose rendered as a
    mesh (MediaPipe gives no mesh of its own), for comparing detection vs retargeting.'''
    import trimesh
    parts = []
    for p in kp:
        s = trimesh.creation.icosphere(subdivisions=1, radius=joint_r)
        s.apply_translation(p)
        parts.append(s)
    for a, b in HAND_EDGES:
        p0, p1 = kp[a], kp[b]
        if np.linalg.norm(p1 - p0) < 1e-6:
            continue
        parts.append(trimesh.creation.cylinder(radius=bone_r, segment=[p0, p1], sections=8))
    mesh = trimesh.util.concatenate(parts)
    mesh.visual.vertex_colors = np.tile(np.array([*color, 255], np.uint8), (len(mesh.vertices), 1))
    return mesh


def palm_canonicalize(kp21, target_len):
    '''Put a raw 21x3 hand (MediaPipe world landmarks, meters) into GeoRT's palm frame
    (+Y thumb, +Z middle, +X normal), origin at palm center, then rescale so the
    wrist->middle-tip length matches the training distribution. Returns [21,3].'''
    kp = np.asarray(kp21, dtype=np.float64)
    pc = kp[[0] + MP_MCP].mean(axis=0)
    z = _n(kp[9] - pc)                 # palm -> middle MCP
    yb = _n(kp[2] - pc)                # toward thumb MCP
    x = _n(np.cross(yb, z))            # palm normal
    y = _n(np.cross(z, x))             # re-orthogonalized thumb direction
    R = np.stack([x, y, z], axis=1)    # columns are the new axes
    local = (kp - pc) @ R
    L = np.linalg.norm(local[12] - local[0]) + 1e-9
    return (local * (target_len / L)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-hand', default='wuji')
    ap.add_argument('-ckpt_tag', default='wuji_mano')
    ap.add_argument('--epoch', type=int, default=0)
    ap.add_argument('--camera', choices=['webcam', 'realsense'], default='webcam')
    ap.add_argument('--device', type=int, default=0, help='webcam device index')
    ap.add_argument('--replay', type=str, default=None,
                    help='no-camera test: loop a bundled dataset (e.g. mano_gesture) through the live path')
    ap.add_argument('--target_scale', type=float, default=DEFAULT_SCALE)
    ap.add_argument('--smooth', type=float, default=0.5, help='EMA factor on qpos (0=frozen,1=no smoothing)')
    ap.add_argument('--mirror', action='store_true', help='mirror the hand (use if the robot is flipped)')
    ap.add_argument('--hand_viz', choices=['mesh', 'skeleton'], default='mesh',
                    help='render the detected hand as a solid capsule model (mesh) or lighter lines (skeleton)')
    ap.add_argument('--show_camera', action='store_true', help='also pop up the annotated camera window')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080)
    args = ap.parse_args()

    config = get_config(args.hand)

    # --- viser scene: robot on -Y, your detected hand on +Y ---
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction('+z')
    wuji_frame = server.scene.add_frame('/wuji', show_axes=False, position=(0.0, -OFFSET, 0.0))
    urdf = ViserUrdf(server, urdf_or_path=resolve_urdf_path(config['urdf_path']), root_node_name='/wuji')
    wuji_frame.position = (0.0, -OFFSET, 0.0)
    urdf_joint_names = urdf.get_actuated_joint_names()
    cfg_index = {n: i for i, n in enumerate(config['joint_order'])}
    perm = np.array([cfg_index[n] for n in urdf_joint_names])
    limits = urdf.get_actuated_joint_limits()
    lo = np.array([(limits[n][0] if limits[n][0] is not None else -np.pi) for n in urdf_joint_names])
    hi = np.array([(limits[n][1] if limits[n][1] is not None else np.pi) for n in urdf_joint_names])
    server.scene.add_label('/lbl_human', 'your hand', position=(0.0, OFFSET, 0.26))
    server.scene.add_label('/lbl_wuji', 'Wuji (retargeted)', position=(0.0, -OFFSET, 0.26))

    status = server.gui.add_text('status', initial_value='starting...', disabled=True)

    # --- model (torch; CPU if no GPU). Loaded lazily so --help etc. stay light. ---
    from geort import load_model
    model = load_model(args.ckpt_tag, epoch=args.epoch)

    state = {'q': None}

    def step(world_kp21):
        '''Retarget one raw 21x3 hand and update the view. Returns qpos or None.'''
        canon = palm_canonicalize(world_kp21, args.target_scale)
        if args.mirror:
            canon = canon * np.array([-1.0, 1.0, 1.0], np.float32)
        q = np.asarray(model.forward(canon), dtype=np.float64)
        if state['q'] is None:
            state['q'] = q
        else:
            state['q'] = args.smooth * q + (1.0 - args.smooth) * state['q']  # EMA
        urdf.update_cfg(np.clip(state['q'][perm], lo, hi))
        # show the camera's detected pose as a hand "model" next to the robot, so you can
        # tell detection errors (the model looks wrong) from retargeting errors (model looks
        # right but the robot doesn't match it).
        if args.hand_viz == 'mesh':
            server.scene.add_mesh_trimesh('/your_hand', skeleton_mesh(canon), position=(0.0, OFFSET, 0.0))
        else:
            segs = np.array([[canon[a], canon[b]] for a, b in HAND_EDGES], dtype=np.float32)
            server.scene.add_line_segments('/your_hand_bones', points=segs,
                                           colors=(255, 120, 80), line_width=4.0, position=(0.0, OFFSET, 0.0))
            server.scene.add_point_cloud('/your_hand_joints', points=canon.astype(np.float32),
                                         colors=np.tile(np.array([[255, 40, 40]], np.uint8), (21, 1)),
                                         point_size=0.008, position=(0.0, OFFSET, 0.0))
        return state['q']

    print(f"\nViser running on port {args.port}. Open the URL above (forward the port if remote).\n")

    # --- no-camera replay mode (test the viewer / offline) ---
    if args.replay:
        data = np.load(get_data_root() / f'{args.replay}.npy')  # [T,21,3]
        print(f"[replay] looping {len(data)} frames of '{args.replay}' through the live path.")
        t = 0
        while True:
            step(data[t % len(data)]); status.value = f"replay {args.replay} frame {t % len(data)}"
            t += 1
            time.sleep(1.0 / 30.0)

    # --- camera mode ---
    from geort.mocap.mediapipe_mocap import MediaPipeHandDetector  # lazy: needs mediapipe
    detector = MediaPipeHandDetector()
    if args.camera == 'realsense':
        from geort.mocap.camera.realsense import RealSenseCamera
        cam = RealSenseCamera()
    else:
        from geort.mocap.camera.webcam import WebcamCamera
        cam = WebcamCamera(device_index=args.device)
    print(f"[camera] {args.camera} started. Show your right hand to the camera.")

    cv2 = None
    if args.show_camera:
        import cv2 as _cv2
        cv2 = _cv2

    while True:
        frame = cam.get_frame().get('rgb')
        if frame is None:
            continue
        res = detector.detect(frame)
        if res['detected'] and len(res['world_coordinates']) == 21:
            step(res['world_coordinates'])
            status.value = 'tracking'
        else:
            status.value = 'no hand detected'
        if cv2 is not None:
            cv2.imshow('camera (q to quit)', res['annotated_img'])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()


if __name__ == '__main__':
    main()
