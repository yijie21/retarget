# Conda environments

By default the pipeline switches between **4 conda envs** (names set in `config/paths.sh`).
You can also run **everything in ONE env** — see [Single-environment setup](#single-environment-setup) below.

## Single-environment setup

The 4-env split exists only because the four upstream repos (SAM3, SAM3D, HaWoR, TAPnet)
ship independent dependency trees — *not* because of a hard incompatibility. They all use
**cu128 + torch 2.7–2.9**, so a single **Python 3.11 / torch 2.8.0+cu128** env runs all of
reconstruction **and** retargeting.

**Build it** (the `sam3d` env is the hard part — it has the from-source CUDA builds; the script
copies it and layers the rest on top):

```bash
# 1. Build the sam3d env first (env/sam3d.yml + the pytorch3d / kaolin / diff-gaussian /
#    geocalib steps in this file). It is torch 2.8.0+cu128 / Python 3.11.
# 2. Then consolidate into one env named `daid`:
./setup/03_single_env.sh            # copies sam3d -> daid, layers sam3 + hawor + tapnet + retargeting
```

What `03_single_env.sh` does (and the gotchas it encodes):
- Copies `sam3d` → `daid` (a plain `cp -a` of the env dir — `conda create --clone` re-solves and
  is pathologically slow on this env).
- Adds the retargeting package (`pip install -e retargeting`; its `pyproject.toml` now allows
  Python ≥ 3.11), TAPIR (torch path; **no JAX needed for tracking**), the SAM3 package + CLIP
  BPE assets, and HaWoR's deps (`smplx`, `torch-scatter` from the pyg pt28 index, `chumpy` from git).
- **The one real conflict to know about:** TAPIR's `jax` pulls **numpy 2.x**, but every CUDA
  extension (pytorch3d/kaolin/geocalib/mujoco-warp/spconv) is built against **numpy 1.26** — so
  the script pins `numpy==1.26.4` and uses a numpy-1.26-compatible `jax==0.4.30`/`chex==0.1.86`.
  (`timm` 0.9.16 vs the SAM3 pin ≥1.0.17 is only a warning — SAM3 runs fine on 0.9.16.)

**Run it** (one env, no per-stage switching). `config/paths.sh` honours pre-set env names:

```bash
conda activate daid
cd reconstruction
ENV_SAM3=daid ENV_SAM3D=daid ENV_HAWOR=daid ENV_TAPNET=daid \
  ./run_pipeline_headless.sh whisking/whisking.mp4 28 knife right     # all stages in `daid`

cd ../retargeting
python launch.py --task whisking --raw-dir ../reconstruction/whisking # same env
```

---

## The 4-environment setup (default)

The pipeline switches between **4 conda envs** (names set in `config/paths.sh`).

**Recommended:** build each env by following its fork's own setup instructions —
[`malik-group/sam3`](https://github.com/malik-group/sam3) (`sam3`),
[`malik-group/sam-3d-objects`](https://github.com/malik-group/sam-3d-objects) (`sam3d`),
[`malik-group/HaWoR`](https://github.com/malik-group/HaWoR) (`hawor`),
[`malik-group/tapnet`](https://github.com/malik-group/tapnet) (`tapnet`).

The two options below are **fallbacks** (e.g. on Blackwell / RTX 50xx, where the forks' cu117/cu121 pins won't run):

- **Build fresh** — `./setup/01_create_envs.sh` (or one at a time: `./setup/01_create_envs.sh sam3|sam3d|hawor|tapnet`). Builds each env from the upstream repos' own dependency files + the recipes in the script. 
- **From the exact pins here** — the `env/*.yml` files are full `conda env export`s of known-working envs, to be treated mainly as a version reference for manual installation.

After the `sam3d` env is set up — whether via your own steps or the commands above —
run this once to un-shadow the repo's `notebook/` package:
```bash
pip uninstall -y notebook   # ensure `notebook.inference` imports cleanly
```

The `sam3d` env also needs GeoCalib for the gravity-estimation step
(`scripts/predict_video_gravity.py`; installed automatically by `01_create_envs.sh`):
```bash
pip install "geocalib @ git+https://github.com/cvg/GeoCalib.git"
```
Note: `env/sam3d.yml` predates this addition — if installing from the exact pins, add geocalib on top.

If SAM 3D inference crashes during GLB/texture baking with `NameError: name 'GaussianRasterizationSettings' is not defined` try below:

```bash
conda activate sam3d
git clone --recursive https://github.com/autonomousvision/mip-splatting.git
cd mip-splatting/submodules/diff-gaussian-rasterization
CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=12.0 FORCE_CUDA=1 python setup.py install
```

If building HaWoR's DROID-SLAM / lietorch fails to compile with
`error: cannot convert 'const at::DeprecatedTypeProperties' to 'c10::ScalarType'` in lietorch's `dispatch.h`,
edit `modules/HaWoR/thirdparty/DROID-SLAM/thirdparty/lietorch/lietorch/include/dispatch.h` and rebuild
(`cd modules/HaWoR/thirdparty/DROID-SLAM && python setup.py install`):
```diff
-    at::ScalarType _st = ::detail::scalar_type(the_type);
+    at::ScalarType _st = the_type.scalarType();
```

| env (default name) | exact-pin YAML | stages | deps from |
|---|---|---|---|
| `sam3`   | `env/sam3.yml`   | 1 — SAM3 segmentation        | `env/sam3.yml` (or refer to original repository) |
| `sam3d`  | `env/sam3d.yml`  | 2, 3, 4 — meshes, pose, opt  | `modules/sam-3d-objects/environments/default.yml` + `requirements*.txt` |
| `hawor`  | `env/hawor.yml`  | 2 — hand reconstruction      | `modules/HaWoR/requirements.txt` (+ torch cu117) |
| `tapnet` | `env/tapnet.yml` | 2.5 — velocity tracking      | `modules/tapnet[torch]` (Python 3.10, torch 2.7 cu128) |


