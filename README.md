# Dexterous Hand Retargeting — a comparison monorepo

Four approaches to **human → robot dexterous-hand retargeting**, gathered in one place
for side-by-side comparison and redevelopment.

| Folder | Method | Paper | What it does |
|---|---|---|---|
| [`toporetarget/`](./toporetarget) | **TopoRetarget** (re-implementation) | [arXiv 2606.16272](https://arxiv.org/abs/2606.16272) | Optimization that preserves **hand–object contact topology** (interaction mesh + Laplacian), MANO → Wuji hand |
| [`GeoRT/`](./GeoRT) | **GeoRT** | [arXiv 2503.07541](https://arxiv.org/abs/2503.07541) | Ultrafast (~1 kHz) **neural** fingertip retargeting via 5 geometric losses; real-time teleop |
| [`spider/`](./spider) | **SPIDER** | [arXiv 2511.09484](https://arxiv.org/abs/2511.09484) | **Physics-informed** retargeting: sampling-based MPC in MuJoCo Warp → dynamically feasible trajectories |
| [`do-as-i-do/`](./do-as-i-do) | **Do As I Do** | [arXiv 2606.19333](https://arxiv.org/abs/2606.19333) | End-to-end **internet video → real dexterous robot**: reconstruction (guided-diffusion tracking) + retargeting (built on SPIDER) |

**The spectrum:** GeoRT preserves *fingertip motion* (fast, object-agnostic) → TopoRetarget preserves
*contact structure* (geometric) → SPIDER preserves *dynamic feasibility* (physics) → Do As I Do wraps a full
*perception + physics* pipeline around SPIDER-style retargeting.

### Conceptual deep-dives (open in a browser)
- `GeoRT/geort_explained.html` — how GeoRT's 5 geometric losses work
- `GeoRT/retargeting_comparison.html` — GeoRT vs TopoRetarget vs SPIDER
- `do-as-i-do/do_as_i_do_analysis.html` — what is actually novel in Do As I Do
- `do-as-i-do/guided_diffusion_tracking.html` — how SAM-3D is turned into a video tracker (training-free)
- `toporetarget/toporetarget_explained.html`, `toporetarget/stage3_laplacian_explained.html`

---

## ⚠️ Before you `git push` — licensing & repo hygiene

This tree contains **license-gated data and ~25 GB of model weights** that **must not** be pushed to a public
remote. A protective [`.gitignore`](./.gitignore) is included that excludes:

- **MANO** body models (`toporetarget/models/`, `**/MANO_*.pkl`) — [license](https://mano.is.tue.mpg.de), not redistributable
- **GRAB / ContactDB** raw data (`toporetarget/grab_raw/`, `GeoRT/data/grab_*`) — not redistributable
- **SAM-3D / HaWoR** weights (`do-as-i-do/reconstruction/weights/`, `.../modules/**/weights/`) — ~25 GB, gated
- **SPIDER** example datasets (`spider/example_datasets/`) — HuggingFace LFS

Keep these ignores. If your remote is private you *may* keep some data, but pushing MANO/GRAB anywhere public
violates their licenses.

### Making it one repo
Each sub-project is **still its own git repo** (`spider/.git`, `GeoRT/.git`, `do-as-i-do/.git`,
`toporetarget/wuji-hand-description/.git`). A plain `git init && git add .` at this level would record them as
empty *submodule pointers*, not their files. Two options:

**A. Flatten into a single monorepo** (simplest for joint redevelopment). This removes the sub-repos' standalone
git history — all four have public origins (below), so history is recoverable by re-cloning.
```bash
cd /home/user2/code/retarget
# remove nested git metadata so the files become normal tracked files
find . -path ./.git -prune -o -name .git -print          # review first
find . -path ./.git -prune -o -name .git -exec rm -rf {} +   # then remove
# do-as-i-do also has submodule gitlinks under reconstruction/modules/*
git init && git add . && git commit -m "Combine 4 retargeting methods for comparison"
```

**B. Keep them as submodules** (preserves history, but each stays a separate repo):
```bash
cd /home/user2/code/retarget && git init
git submodule add https://github.com/yijie21/GeoRT GeoRT
git submodule add https://github.com/facebookresearch/spider spider
git submodule add https://github.com/malik-group/do-as-i-do do-as-i-do
# toporetarget/ is your own work → just `git add toporetarget`
```

Upstream origins: GeoRT → `github.com/yijie21/GeoRT` · SPIDER → `github.com/facebookresearch/spider` ·
Do As I Do → `github.com/malik-group/do-as-i-do`.

> **Note:** the sub-projects were moved here from `/home/user2/code/`. Any `pip install -e .` editable installs
> done before the move point at the old paths — re-run `pip install -e .` inside the affected conda env after moving.

---

## How to run each

### 1. `toporetarget/` — TopoRetarget re-implementation ✅ runs (CPU)
Synthetic grasp + the 4-loss ablation, with an interactive 3-D viewer.
```bash
cd toporetarget/toporetarget_impl
python run_retarget.py                 # synthetic cylinder grasp (full method + ablations)
python run_retarget.py --object sphere
xdg-open viewer/index.html             # orbit / scrub / toggle each loss term
```
Real GRAB data needs a one-time `manoconv` conda env (MANO/chumpy need Python <3.11) to extract keypoints —
see [`toporetarget/toporetarget_impl/README.md`](./toporetarget/toporetarget_impl/README.md). Deps: torch-cpu,
pytorch_kinematics, trimesh, scipy, numpy.

### 2. `GeoRT/` — ultrafast neural retargeting ✅ runs (demos need no torch)
```bash
cd GeoRT
pip install -e .
pip install numpy viser yourdfpy
# Side-by-side: human MANO hand vs retargeted Wuji hand on a real grasp (replays precomputed angles, no GPU/torch)
python geort/mocap/compare_viser.py -hand wuji -ckpt_tag wuji_mano -data grab_mug_drink
python geort/mocap/compare_viser.py -hand wuji -ckpt_tag wuji_mano -data furelise_demo
# Train your own (needs torch + SAPIEN 2.x):
python geort/trainer.py -hand allegro_right -human_data human_alex -ckpt_tag geort_1
```
See [`GeoRT/README.md`](./GeoRT/README.md). Requires **SAPIEN 2.x** (not 3.x) for training/visualization.

### 3. `spider/` — physics-informed retargeting ✅ runs (needs CUDA GPU)
```bash
cd spider
conda create -n spider python=3.12 && conda activate spider
pip install -r requirements.txt && pip install --no-deps -e .
# example datasets (HuggingFace LFS):
git lfs install && git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets
python examples/run_mjwp.py                                   # run on a processed trial
# faster Hydra-configured runs:
uv run examples/run_mjwp_fast.py +override=oakinkv2_fast task=pick_spoon_bowl embodiment_type=right robot_type=xhand
```
See [`spider/README.md`](./spider/README.md). MuJoCo-Warp physics runs on GPU.

### 4. `do-as-i-do/` — video → real robot ⚠️ partial setup (see below)
The **retargeting** half is built and runs on GPU; the **reconstruction** half is not fully built here.

---

## `do-as-i-do` setup status & instructions

Do As I Do is a 3-stage pipeline: **reconstruction** (RGB video → hand+object 3-D + 6-DoF pose) →
**retargeting** (→ physics-feasible robot trajectory, built on SPIDER) → **deployment** (real UR3e + Sharpa hands).

### What is already set up on this machine
- ✅ All 5 submodules checked out (incl. HaWoR's nested `lietorch`/`eigen`/`DROID-SLAM`).
- ✅ ~25 GB of weights downloaded (SAM-3D 13 GB, HaWoR ckpts, TAPIR, Metric3D).
- ✅ **MANO** copied into HaWoR (`modules/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl`, …).
- ✅ Conda envs **`retargeting`** (MuJoCo-Warp runs on the GPU), **`sam3`**, **`tapnet`** — all torch cu128, CUDA OK.
- ✅ Headless segmentation driver `reconstruction/run_pipeline_headless.sh` (object via text prompt, no X display).

### What is NOT done (the blockers)
- ❌ Conda envs **`sam3d`** and **`hawor`** have no torch/pip deps — they need **from-source CUDA builds for
  Blackwell `sm_120`**: `pytorch3d`, `lietorch`, `diff_gaussian_rasterization`, `DROID-SLAM`. The provided cu128
  env-exports fail at the pip block; the original build script pins cu121/cu117 (wrong for this GPU).
- ⚠️ **VRAM:** reconstruction is spec'd for **≥32 GB**; this GPU has **16 GB**, so the reconstruction run may OOM
  even after the envs build (SAM-3D alone loads ~11.6 GB of generator weights).

### To finish the build (resume here)
For **each** of `sam3d` and `hawor`:
```bash
conda activate sam3d   # (and later: hawor)
# 1) install matching torch FIRST so from-source builds can see it
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# 2) install each repo's requirements with the torch* pins stripped (they pin cu121/cu117):
grep -viE '^torch|^torchvision|^torchaudio' reconstruction/modules/sam-3d-objects/requirements.txt | pip install -r /dev/stdin
# 3) build the from-source CUDA extensions with build isolation OFF:
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# hawor also needs: the lietorch dispatch.h one-line patch for torch>=2.x
#   (::detail::scalar_type(t) -> t.scalarType()), then build DROID-SLAM:
( cd reconstruction/modules/HaWoR/thirdparty/DROID-SLAM && python setup.py install )
# sam3d also needs the Mip-Splatting diff_gaussian_rasterization (inria backend) built — see
#   reconstruction/env/README.md "Stage-2 manual steps".
pip install -e reconstruction/modules/sam-3d-objects   # expose sam3d_objects pkg
```
Full per-env recipes are in [`do-as-i-do/reconstruction/env/README.md`](./do-as-i-do/reconstruction/env/README.md).

### To run it once the envs build
```bash
cd do-as-i-do/reconstruction
# headless: object segmented by text prompt instead of an interactive click
./run_pipeline_headless.sh whisking/whisking.mp4 125 whisk right
#   stages: SAM3 seg → SAM-3D mesh → MoGe pointmaps → HaWoR hands → TAPIR velocity
#           → guided-diffusion object tracking → gravity/scale optimize

cd ../retargeting
pip install -e .                                          # re-run after the move (editable path changed)
python launch.py --task whisking --raw-dir ../reconstruction/whisking
#   → opens a viser web UI of the Sharpa robot hand reproducing the whisking interaction
```
The reconstruction needs an `hf auth login` with access to the gated `facebook/sam3` + `facebook/sam-3d-objects`.

---

## Layout
```
retarget/
├── README.md            ← this file
├── .gitignore           ← protects gated data / weights from being pushed
├── toporetarget/        ← TopoRetarget re-implementation (your work) + GRAB/MANO data + explainer HTMLs
│   ├── toporetarget_impl/   the method + viewer
│   ├── grab_raw/  models/   (gitignored: GRAB + MANO)
│   └── wuji-hand-description/
├── GeoRT/               ← github.com/yijie21/GeoRT
├── spider/              ← github.com/facebookresearch/spider
└── do-as-i-do/          ← github.com/malik-group/do-as-i-do
```
