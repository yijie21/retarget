"""
python demo.py --video_path /path/to/video.mp4 --vis_mode cam --img_focal <fx> --static_camera
"""

import argparse
import sys
import os

import torch
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import joblib
import subprocess
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='example/video_0.mp4')
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint",  type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight",  type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--vis_mode",  type=str, default='cam', help='cam | world')
    parser.add_argument("--static_camera", action='store_true', default=False)
    args = parser.parse_args()

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    if not args.static_camera:
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            hawor_slam(args, start_idx, end_idx)
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)
    else:
        R_w2c_sla_all = torch.eye(3).repeat(len(imgfiles), 1, 1)
        t_w2c_sla_all = torch.zeros(len(imgfiles), 3)
        R_c2w_sla_all = torch.eye(3).repeat(len(imgfiles), 1, 1)
        t_c2w_sla_all = torch.zeros(len(imgfiles), 3)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all, args.static_camera)

    # vis sequence for this video
    hand2idx = {
        "right": 1,
        "left": 0
    }
    vis_start = 0
    # vis_end = pred_trans.shape[1] - 1
    vis_end = pred_trans.shape[1]
            
    # get faces
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)

    # get right hand vertices and joints
    hand = 'right'
    hand_idx = hand2idx[hand]
    pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    right_verts = pred_glob_r['vertices'][0]
    right_joints = pred_glob_r['joints'][0]  # (T, 21, 3)
    right_dict = {
            'vertices': right_verts.unsqueeze(0),
            'faces': faces_right,
        }

    # get left hand vertices and joints
    faces_left = faces_right[:,[0,2,1]]
    hand = 'left'
    hand_idx = hand2idx[hand]
    pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    left_verts = pred_glob_l['vertices'][0]
    left_joints = pred_glob_l['joints'][0]  # (T, 21, 3)
    left_dict = {
            'vertices': left_verts.unsqueeze(0),
            'faces': faces_left,
        }

    R_x = torch.tensor([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())
    right_joints = torch.einsum('ij,tnj->tni', R_x, right_joints.cpu())
    left_joints = torch.einsum('ij,tnj->tni', R_x, left_joints.cpu())

    # Dump left and right hand vertices, joints, and faces into a single numpy file as a dict
    left_vertices_np = (torch.einsum('bij,bvj->bvi', R_w2c_sla_all[vis_start:vis_end], left_dict['vertices'].squeeze()) + t_w2c_sla_all[vis_start:vis_end, None]).cpu().numpy()
    left_faces_np = left_dict['faces'].astype(np.int32)
    right_vertices_np = (torch.einsum('bij,bvj->bvi', R_w2c_sla_all[vis_start:vis_end], right_dict['vertices'].squeeze()) + t_w2c_sla_all[vis_start:vis_end, None]).cpu().numpy()
    right_faces_np = right_dict['faces'].astype(np.int32)
    right_joints_np = (torch.einsum('bij,bvj->bvi', R_w2c_sla_all[vis_start:vis_end], right_joints[vis_start:vis_end]) + t_w2c_sla_all[vis_start:vis_end, None]).cpu().numpy()
    left_joints_np = (torch.einsum('bij,bvj->bvi', R_w2c_sla_all[vis_start:vis_end], left_joints[vis_start:vis_end]) + t_w2c_sla_all[vis_start:vis_end, None]).cpu().numpy()

    mesh_data = {
        'left_vertices': left_vertices_np,
        'left_faces': left_faces_np,
        'left_joints': left_joints_np,
        'right_vertices': right_vertices_np,
        'right_faces': right_faces_np,
        'right_joints': right_joints_np,
        'left_trans': pred_trans[0, vis_start:vis_end].cpu().numpy(),
        'left_rot': pred_rot[0, vis_start:vis_end].cpu().numpy(),
        'left_hand_pose': pred_hand_pose[0, vis_start:vis_end].cpu().numpy(),
        'left_betas': pred_betas[0, vis_start:vis_end].cpu().numpy(),
        'left_valid': pred_valid[0, vis_start:vis_end],
        'right_trans': pred_trans[1, vis_start:vis_end].cpu().numpy(),
        'right_rot': pred_rot[1, vis_start:vis_end].cpu().numpy(),
        'right_hand_pose': pred_hand_pose[1, vis_start:vis_end].cpu().numpy(),
        'right_betas': pred_betas[1, vis_start:vis_end].cpu().numpy(),
        'right_valid': pred_valid[1, vis_start:vis_end],
    }
    np.savez(os.path.join(seq_folder, "all_hand_meshes.npz"), **mesh_data)

    with open(os.path.join(seq_folder, "img_focal.txt"), "w") as f:
        f.write(str(img_focal))

    # Here we use aitviewer(https://github.com/eth-ait/aitviewer) for simple visualization.
    if args.vis_mode == 'world': 
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        run_vis2_on_video(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w=R_c2w_sla_all[vis_start:vis_end], t_c2w=t_c2w_sla_all[vis_start:vis_end])
    elif args.vis_mode == 'cam':
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        run_vis2_on_video_cam(left_dict, right_dict, output_pth, img_focal, image_names, R_w2c=R_w2c_sla_all[vis_start:vis_end], t_w2c=t_w2c_sla_all[vis_start:vis_end])

    # Save aitviewer output pngs as a video (overlay.mp4) using ffmpeg
    aitviewer_images_dir = os.path.join(output_pth, "aitviewer", "images", "rgb")
    mp4_path = os.path.join(output_pth, "overlay.mp4")

    # Input pattern for pngs: e.g., .../images/rgb/0000.png, 0001.png, etc.
    input_pattern = os.path.join(aitviewer_images_dir, "%04d.png")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-framerate", "30",
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        mp4_path,
    ]

    print("Saving aitviewer rgb images as a video overlay.mp4 with ffmpeg...")
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video saved as: {mp4_path}")
    except Exception as e:
        print(f"Error running ffmpeg: {e}")

    print("finish")
