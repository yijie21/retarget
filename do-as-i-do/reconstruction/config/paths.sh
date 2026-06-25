# ───────────────────────── reconstruction pipeline paths ─────────────────────────
# Sourced by run_pipeline.sh; the Python scripts read these via os.environ

# Root of this extracted project 
RECON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RECON_ROOT

# ── Modules ──
export SAM3D_DIR="$RECON_ROOT/modules/sam-3d-objects"
export FASTSAM3D_DIR="$RECON_ROOT/modules/Fast-SAM3D"
export HAWOR_DIR="$RECON_ROOT/modules/HaWoR"
export TAPNET_DIR="$RECON_ROOT/modules/tapnet"
export SAM3_PKG_DIR="$RECON_ROOT/modules/sam3"

export SCRIPTS_DIR="$RECON_ROOT/scripts"

# ── Which repo root the MoGe pointmap script imports from ──
export SAM3D_REPO_ROOT="$FASTSAM3D_DIR"         

# ── Weights (populated by setup/02_fetch_weights.sh) ──
export WEIGHTS_DIR="$RECON_ROOT/weights"
export TAPNET_CKPT="$WEIGHTS_DIR/tapnet/bootstapir_checkpoint_v2.pt"

# ── Conda env names ──
# Defaults match what `setup/01_create_envs.sh` creates. These respect values
# pre-set in the environment, so you can run every stage in ONE consolidated env:
#   export ENV_SAM3=daid ENV_SAM3D=daid ENV_HAWOR=daid ENV_TAPNET=daid
# (see env/README.md "Single-environment setup"). If you are reusing pre-existing
# local envs with different names, set these accordingly.
export ENV_SAM3="${ENV_SAM3:-sam3}"
export ENV_SAM3D="${ENV_SAM3D:-sam3d}"
export ENV_HAWOR="${ENV_HAWOR:-hawor}"
export ENV_TAPNET="${ENV_TAPNET:-tapnet}"

# ── Host / GPU ──
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# X display used by the click-based SAM3 segmentation UI (Stage 1).
export SAM3_DISPLAY=:1
