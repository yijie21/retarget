#!/usr/bin/env python
"""Render a side-by-side comparison video: original demo video (left) vs. the
retargeted robot-hand + object trajectory (right).

Consumes a retargeting output dir (``scene.xml`` + ``trajectory_mjwp.npz``) and the
original demo clip, renders the optimized trajectory headlessly with MuJoCo (EGL),
and composes the two panels with ffmpeg (labels + length-matched).

Examples
--------
# explicit paths
python render_comparison.py \
    --output-dir outputs/sharpa/right/ball_release/0 \
    --orig-video ../reconstruction/runs/ball_release/ball_release.mp4

# or let it derive paths from the task (robot=sharpa, hand=right by default)
python render_comparison.py --task ball_release \
    --raw-dir ../reconstruction/runs/ball_release

Headless rendering: this script sets ``MUJOCO_GL=egl`` and auto-selects a working
EGL device (some hosts expose several EGL devices and only the GPU ones can make a
desktop-GL context). Override with ``MUJOCO_EGL_DEVICE_ID`` if needed.
"""
import argparse
import os
import subprocess
import sys

# Headless GL must be configured BEFORE importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("PYOPENGL_PLATFORM", None)  # let mujoco set it to egl
# A conda libEGL on LD_LIBRARY_PATH can shadow the working system/NVIDIA EGL.
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if p and "envs" not in p  # drop conda-env lib dirs
)


def _make_renderer(model, height, width):
    """Build a mujoco.Renderer, probing EGL device ids until one works."""
    import mujoco

    if "MUJOCO_EGL_DEVICE_ID" in os.environ:
        return mujoco.Renderer(model, height=height, width=width)
    last = None
    for dev in range(8):
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(dev)
        try:
            r = mujoco.Renderer(model, height=height, width=width)
            # force a context-creating op to confirm the device really works
            d = mujoco.MjData(model)
            mujoco.mj_forward(model, d)
            r.update_scene(d)
            r.render()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    raise RuntimeError(f"No working EGL device found (last error: {last})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", help="retargeting output dir with scene.xml + trajectory_mjwp.npz")
    ap.add_argument("--task", help="video/task name (used to derive --output-dir if not given)")
    ap.add_argument("--robot", default="sharpa")
    ap.add_argument("--hand", default="right")
    ap.add_argument("--raw-dir", help="reconstruction output dir (used to derive --orig-video)")
    ap.add_argument("--orig-video", help="original demo clip (left panel)")
    ap.add_argument("--out", help="output mp4 path (default: <output-dir>/comparison_<task>.mp4)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--azimuth", type=float, default=138.0)
    ap.add_argument("--elevation", type=float, default=-16.0)
    ap.add_argument("--zoom", type=float, default=1.4, help="distance margin; larger = more zoomed out")
    ap.add_argument("--trim-warmup", action="store_true",
                    help="skip the optimizer warmup prefix so the right panel starts at the manipulation")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.output_dir
    if out_dir is None:
        if not args.task:
            ap.error("provide either --output-dir or --task")
        out_dir = os.path.join(here, "outputs", args.robot, args.hand, args.task, "0")
    out_dir = os.path.abspath(out_dir)
    task = args.task or os.path.basename(os.path.dirname(out_dir))

    scene = os.path.join(out_dir, "scene.xml")
    traj = os.path.join(out_dir, "trajectory_mjwp.npz")
    for p in (scene, traj):
        if not os.path.exists(p):
            ap.error(f"missing required file: {p}")

    orig = args.orig_video
    if orig is None and args.raw_dir:
        raw = os.path.abspath(args.raw_dir)
        cands = [os.path.join(raw, f"{task}.mp4"), os.path.join(raw, task, f"{task}.mp4")]
        orig = next((c for c in cands if os.path.exists(c)), None)
    if not orig or not os.path.exists(orig):
        ap.error("could not find the original demo video; pass --orig-video explicitly")
    orig = os.path.abspath(orig)

    out_path = args.out or os.path.join(out_dir, f"comparison_{task}.mp4")

    import numpy as np
    import mujoco
    import imageio.v2 as imageio

    m = mujoco.MjModel.from_xml_path(scene)
    # hide reference-target marker bodies (ref_*) so only the real robot+object show
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm.startswith("ref_"):
            m.geom_rgba[g, 3] = 0.0
    # bodies to keep framed: everything except world(0) and hidden ref_* markers
    key_bodies = [b for b in range(1, m.nbody)
                  if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("ref_")]

    data = mujoco.MjData(m)
    d = np.load(traj)
    Q = d["qpos"].reshape(-1, m.nq)
    T = d["time"].reshape(-1)
    sim_dt = float(np.median(np.diff(T)))
    if args.trim_warmup and "warmup_progress" in d.files:
        # windows are (n_win, win_len, nq); drop windows still in warmup
        wp = d["warmup_progress"]
        win_len = d["qpos"].shape[1]
        first_real = int(np.argmax(wp >= 1.0)) if np.any(wp >= 1.0) else 0
        Q = Q[first_real * win_len:]
    dur = (len(Q) - 1) * sim_dt
    n_out = max(1, int(round(dur * args.fps)))
    idxs = [min(len(Q) - 1, int(round((fi / args.fps) / sim_dt))) for fi in range(n_out)]

    # per-frame scene centroid + radius -> smoothed lookat + auto fit distance
    cents, radii = [], []
    for idx in idxs:
        data.qpos[:] = Q[idx]
        mujoco.mj_forward(m, data)
        pts = data.xpos[key_bodies]
        c = pts.mean(0)
        cents.append(c)
        radii.append(np.linalg.norm(pts - c, axis=1).max())
    cents = np.array(cents)
    k = max(1, int(round(0.5 * args.fps)))  # ~0.5s smoothing
    pad = np.pad(cents, ((k // 2, k // 2), (0, 0)), mode="edge")
    look = np.stack([np.convolve(pad[:, c], np.ones(k) / k, mode="valid")
                     for c in range(3)], 1)[:n_out]
    fovy = np.deg2rad(m.vis.global_.fovy if m.vis.global_.fovy > 0 else 45.0)
    distance = float(max(radii)) / np.tan(fovy / 2.0) * args.zoom

    r = _make_renderer(m, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation

    frames_dir = os.path.join(out_dir, "_robot_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for fi, idx in enumerate(idxs):
        data.qpos[:] = Q[idx]
        mujoco.mj_forward(m, data)
        cam.lookat[:] = look[fi]
        r.update_scene(data, camera=cam)
        imageio.imwrite(os.path.join(frames_dir, f"{fi:04d}.png"), r.render())
    print(f"[render] {n_out} robot frames ({dur:.2f}s @ {args.fps}fps), distance={distance:.2f}")

    # --- compose side-by-side with ffmpeg ---
    font = next((f for f in [
        os.path.join(sys.prefix, "fonts", "DejaVuSans.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ] if os.path.exists(f)), None)

    def label(text, color, boxcolor):
        if not font:
            return ""
        return (f",drawtext=fontfile={font}:text={text}:x=(w-text_w)/2:y=14:"
                f"fontsize=26:fontcolor={color}:box=1:boxcolor={boxcolor}:boxborderw=8")

    robot_dur = n_out / args.fps
    filt = (
        f"[0:v]fps={args.fps},setpts=PTS-STARTPTS,scale=-2:{args.height},"
        f"tpad=stop_mode=clone:stop_duration=20,trim=duration={robot_dur:.3f}"
        f"{label('Original demo', 'white', 'black@0.5')}[L];"
        f"[1:v]setpts=PTS-STARTPTS{label('Retargeted - ' + args.robot + ' hand', 'black', 'white@0.65')}[R];"
        f"[L][R]hstack=inputs=2[v]"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", orig,
        "-framerate", str(args.fps), "-i", os.path.join(frames_dir, "%04d.png"),
        "-filter_complex", filt, "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-r", str(args.fps), "-movflags", "+faststart", out_path,
    ]
    subprocess.run(cmd, check=True)
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)  # drop temp render frames
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
