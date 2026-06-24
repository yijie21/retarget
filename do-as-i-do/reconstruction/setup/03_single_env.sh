#!/bin/bash
# [03] Build ONE consolidated conda env ("daid") that runs the whole pipeline —
# all of reconstruction (sam3 + sam3d + hawor + tapnet) AND retargeting — instead
# of 4+1 separate envs. See env/README.md "Single-environment setup".
#
# Strategy: the `sam3d` env is the hard part (torch 2.8.0+cu128 / Python 3.11 with
# pytorch3d, kaolin, nvdiffrast, diff-gaussian-rasterization, moge, utils3d, spconv,
# geocalib all built from source for sm_120). We start from a copy of it and layer
# the other stages' deps on top — they're all torch 2.7-2.9/cu128, so they coexist.
#
# Prereq: a working `sam3d` env (build it via env/sam3d.yml + the pytorch3d/kaolin/
# diff-gaussian steps in env/README.md). Then:  ./setup/03_single_env.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
SRC_ENV="${SRC_ENV:-sam3d}"          # base env to copy (has the CUDA builds)
DST_ENV="${DST_ENV:-daid}"
REPO="$(cd "$ROOT/.." && pwd)"        # monorepo root (has retargeting/)
PYG="https://data.pyg.org/whl/torch-2.8.0+cu128.html"

# 1. Copy the sam3d base (fast; conda --clone re-solves and is very slow on this env).
SRC_PREFIX="$(conda env list | awk -v e="$SRC_ENV" '$1==e{print $NF}')"
DST_PREFIX="$(dirname "$SRC_PREFIX")/$DST_ENV"
echo "[03] copying $SRC_PREFIX -> $DST_PREFIX"
[ -d "$DST_PREFIX" ] || cp -a "$SRC_PREFIX" "$DST_PREFIX"
conda activate "$DST_ENV"
export CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX"
pip(){ python -m pip "$@"; }

# 2. Retargeting (needs requires-python >=3.11 — already set in pyproject.toml).
pip install -e "$REPO/retargeting"

# 3. TAPIR (torch path) + its non-jax media deps + the few deps `--no-deps` skips.
pip install -e "$ROOT/modules/tapnet" --no-deps
pip install mediapy ffmpeg-python einshape dm-tree chex

# 4. SAM3 segmentation package + deps + CLIP BPE assets.
pip install -e "$ROOT/modules/sam3" --no-deps
pip install fairscale fvcore yacs scikit-image numba submitit regex
SITE="$(python -c 'import site;print(site.getsitepackages()[0])')"
mkdir -p "$SITE/assets" && cp -rn "$ROOT/modules/sam3/assets/." "$SITE/assets/" || true

# 5. HaWoR deps (HaWoR itself runs from its dir). torch-scatter from the pyg pt28 index;
#    chumpy from git (PyPI maxes at 0.70). SLAM deps (lietorch/droid) are NOT needed
#    because demo.py is run with --static_camera.
pip install smplx ultralytics ultralytics-thop supervision lapx roma \
            pytorch-minimize mmengine hydra-colorlog webdataset natsort
pip install torch-scatter -f "$PYG"
pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy.git"

# 6. Pin the cross-stage versions: TAPIR's jax wants numpy 2.x, but every CUDA ext
#    (pytorch3d/kaolin/geocalib/mujoco-warp/spconv) was built against numpy 1.26 —
#    keep numpy 1.26 and use a numpy-1.26-compatible jax/chex. (THE key conflict.)
pip install "numpy==1.26.4" "jax==0.4.30" "jaxlib==0.4.30" "chex==0.1.86" "ml-dtypes==0.4.1"

# 7. Same runtime fix as the 4-env setup: Jupyter's `notebook` shadows SAM3D's local
#    notebook/ package used by generate_mesh_sam3d.py.
pip uninstall -y notebook notebook-shim 2>/dev/null || true

echo "[03] Done. Single env '$DST_ENV' ready."
echo "    Reconstruction:  ENV_SAM3=$DST_ENV ENV_SAM3D=$DST_ENV ENV_HAWOR=$DST_ENV ENV_TAPNET=$DST_ENV \\"
echo "                     ./run_pipeline_headless.sh VIDEO 28 knife right"
echo "    Retargeting:     conda activate $DST_ENV; cd ../retargeting; python launch.py --task ... --raw-dir ..."
