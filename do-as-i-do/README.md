<img width="2563" height="742" alt="Frame 261" src="https://github.com/user-attachments/assets/fc36b68d-80af-4c8d-ab09-1e0e44a2193e" />

# Do as I Do

[**Project Page**](https://do-as-i-do.com/) | [**arXiv**](https://arxiv.org/abs/2606.19333) 

Code release for Do as I Do.

Each part of our pipeline is contained in its own folder. External code references are provided as git submodules with our changes baked in.

- **`reconstruction/`** — object + hand reconstruction and 6-DoF pose tracking from a hand-object
  demo video (SAM3 → SAM3D mesh → MoGe pointmaps → HaWoR → TAPIR → guided diffusion for tracking → (optionally) projection).
  Full details in [`reconstruction/README.md`](reconstruction/README.md).
- **`retargeting/`** — retargets the reconstructed hand-object demo onto a robot hand
  (dataset processing → convex decomposition → MJCF scene generation → IK → sampling-based MPC
  in MuJoCo Warp). Consumes the reconstruction pipeline's output directly.
  Full details in [`retargeting/README.md`](retargeting/README.md).
- **`deployment/`** — replay a retargeted demo on the real robot: a barebones
  MuJoCo replay/IK pass turns a retargeting output into a dual-UR3e joint
  trajectory, which is then streamed to the UR3e arms + Sharpa Wave hands.
  Full details in [`deployment/README.md`](deployment/README.md).

## Run the full pipeline (reconstruction → retargeting)

Given a hand-object demo video, go from raw clip to a retargeted robot-hand trajectory in three steps.
Requires an NVIDIA GPU (≥32 GB VRAM), HuggingFace access to `facebook/sam3` + `facebook/sam-3d-objects`,
and a MANO download. See the per-folder READMEs for full details.

**1. Set up the environment(s).** Either one consolidated conda env (simplest) or the default four:

```bash
# ONE env (Python 3.11 / torch 2.8) — runs reconstruction AND retargeting:
cd reconstruction && ./setup/03_single_env.sh && conda activate daid    # builds `daid`
./setup/02_fetch_weights.sh --download                                  # weights (needs HF auth + MANO)
# (or the default 4-env setup — see reconstruction/env/README.md)
```

**2. Reconstruct** the object mesh + 6-DoF object pose + hand tracks from the video. The output is
written next to the video and becomes retargeting's input:

```bash
# args: VIDEO  REF_FRAME  OBJECT  ANCHOR_HAND       (single env shown; drop the ENV_* vars for the 4-env setup)
ENV_SAM3=daid ENV_SAM3D=daid ENV_HAWOR=daid ENV_TAPNET=daid \
  ./run_pipeline_headless.sh whisking/whisking.mp4 28 whisk right
# → whisking/  with config.json, gravity.json, all_hand_meshes.npz, the object .obj,
#   and obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json
```

**3. Retarget** onto the robot hand (physics-optimized in MuJoCo Warp); opens a viser web viewer:

```bash
cd ../retargeting
python launch.py --task whisking --raw-dir ../reconstruction/whisking   # add --no-show-viewer for headless
# → outputs/<robot>/<hand>/<task>/0/trajectory_mjwp.npz  (the optimized robot-hand + object trajectory)
```

**4. (Optional) Export a side-by-side comparison video** — original demo (left) vs. the
retargeted robot hand + object (right), rendered headlessly from the optimized trajectory:

```bash
# still in retargeting/
python render_comparison.py --task whisking --raw-dir ../reconstruction/whisking
# → outputs/<robot>/<hand>/<task>/0/comparison_<task>.mp4
```

That `trajectory_mjwp.npz` is the input to [`deployment/`](deployment/README.md) for real-robot replay.

