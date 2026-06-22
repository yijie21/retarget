import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import argparse
import numpy as np
from glob import glob
from lib.pipeline.tools import detect_track
from natsort import natsorted
import subprocess


def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    command = [
        'ffmpeg',
        '-i', video_path,
        '-vsync', '0',
        '-start_number', '0',
        os.path.join(output_folder, '%04d.png')
    ]

    subprocess.run(command, check=True)


def resolve_paths(video_path):
    """Resolve seq_folder and img_folder from a video file or a directory of frames."""
    if os.path.isdir(video_path):
        img_folder = os.path.abspath(video_path)
        seq_folder = os.path.dirname(img_folder)
    else:
        root = os.path.dirname(video_path)
        seq = os.path.basename(video_path).split('.')[0]
        seq_folder = os.path.join(root, seq)
        img_folder = os.path.join(seq_folder, 'extracted_images')
    return seq_folder, img_folder


def detect_track_video(args):
    seq_folder, img_folder = resolve_paths(args.video_path)
    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)
    print(f'Running detect_track on {args.video_path} ...')

    ##### Extract Frames #####
    imgfiles = natsorted(glob(f'{img_folder}/*.png'))
    if len(imgfiles) > 0:
        print("Skip extracting frames")
    else:
        _ = extract_frames(args.video_path, img_folder)
    imgfiles = natsorted(glob(f'{img_folder}/*.png'))

    ##### Detection + Track #####
    print('Detect and Track ...')

    start_idx = 0
    end_idx = len(imgfiles)

    if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy'):
        print(f"skip track for {start_idx}_{end_idx}")
        return start_idx, end_idx, seq_folder, imgfiles
    os.makedirs(f"{seq_folder}/tracks_{start_idx}_{end_idx}", exist_ok=True)
    boxes_, tracks_ = detect_track(imgfiles, thresh=0.2)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy', boxes_)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', tracks_)

    return start_idx, end_idx, seq_folder, imgfiles

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--input_type", type=str, default='file')
    args = parser.parse_args()

    detect_track_video(args)