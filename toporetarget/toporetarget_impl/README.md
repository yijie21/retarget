# TopoRetarget (re)implementation — MANO hand-object → Wuji robot hand

A from-scratch implementation of the **optimization stage** of *TopoRetarget:
Interaction-Preserving Retargeting for Dexterous Manipulation* (arXiv 2606.16272),
applied to the **Wuji Hand** URDF.

Given a human (MANO-topology) hand-object interaction, it solves for the robot
base pose + joint angles that reproduce the **same interaction** — preserving the
relative hand-object geometry while avoiding penetration — and ships an
**interactive 3-D viewer** to scrub the optimization and toggle each loss term to
see its effect (the ablation study).

![pipeline] (open `viewer/index.html`)

---

## Quick start

```bash
cd toporetarget_impl
# (deps already installed: torch-cpu, pytorch_kinematics, trimesh, scipy, numpy)
python run_retarget.py                 # synthetic cylinder grasp (default)
python run_retarget.py --object sphere # synthetic sphere grasp
xdg-open viewer/index.html             # orbit / scrub / toggle losses
```

`run_retarget.py` runs the **full method + 4 ablations** and writes everything the
viewer needs to `viewer/data.js`.

---

## What maps to what in the paper

| Paper | Here | File |
|---|---|---|
| Stage 1 — relative bone-direction init `E_bone` | `e_bone`, Procrustes warm-start | `optimize.py`, `robot.py` |
| Stage 2 — interaction mesh (Delaunay, shared edges) | `build_edges` | `interaction.py` |
| Stage 3 — distance-aware weights + Laplacian coords `E_IM` | `laplacian_operator`, `e_im` | `interaction.py`, `optimize.py` |
| Stage 4 — `E_reg` + penetration (slack-like) | `w_reg_term`, `e_pen` | `optimize.py` |
| Full per-frame constrained objective | `retarget(...)` | `optimize.py` |

The objective minimized (positional terms in mm² so weights stay O(1)):

```
L =  w_IM·E_IM  +  w_bone·E_bone  +  w_reg·E_reg  +  w_pen·E_pen  +  w_lim·E_lim
```

Penetration is the differentiable, slack-like hinge `mean(relu(-phi)²)` where `phi`
is the signed distance (>0 outside). Joint limits are enforced by a barrier **and**
projected clamping each step. Base orientation uses the continuous 6-D rotation
representation; the base is initialised by Kabsch alignment of the zero-pose robot
keypoints to the human keypoints.

### Keypoint correspondence (MANO 21 ↔ Wuji)
`robot.py::_keypoint_frames()` maps MANO order
`[wrist, thumb×4, index×4, middle×4, ring×4, pinky×4]` to Wuji frames
`palm` and per-finger `link2/link3/link4/tip` (finger1 = thumb).

---

## Ablations (the viewer's main feature)

Each run sets one loss weight to 0. On the synthetic tall-cylinder grasp:

| run | contact | max-pen | takeaway |
|---|---|---|---|
| **full** | ~3.3 mm | ~0.8 mm | clean human-like side grasp |
| **no E_IM** | ~119 mm | 0 mm | nothing pulls the hand to the object → never contacts |
| **no penetration** | ~21 mm | ~39 mm | fingers sink through the object |
| **no E_bone** | ~5 mm | ~0.8 mm | contact ok, articulation less human-like |
| **no E_reg** | ~3.4 mm | ~0.8 mm | tighter on one frame, less smooth over a sequence |

In the viewer: pick a run (top), scrub the optimization (bottom), watch the loss
curves (right) and the live contact / max-penetration metrics (top-right). Toggle
the **interaction mesh** overlay (left) to see the Delaunay edges colour-coded
hand-hand / object-object / cross (the cross edges carry the interaction).

---

## Real GRAB data (working pipeline)

GRAB + the MANO model are license-gated; this repo was run against subject **s1**
of GRAB plus `mano_v1_2` and the ContactDB object meshes. Because MANO/chumpy need
old Python, hand-joint extraction runs in a separate **conda env** and dumps a tiny
per-frame `.npz`; the main (py3.13/torch) pipeline then consumes that.

```bash
# one-time: conda env for MANO forward (chumpy needs python<3.11)
conda create -y -n manoconv python=3.10 numpy=1.23 scipy
conda activate manoconv && pip install "setuptools<65" chumpy trimesh smplx \
    torch --index-url https://download.pytorch.org/whl/cpu

# 1) extract 21 MANO keypoints + object pose for one frame (conda env)
conda run -n manoconv python grab_extract.py \
    --seq ../grab_raw/s1_data/s1/cylindersmall_lift.npz --frame 1034 \
    --out grab_frames/cylindersmall_lift_f1034.npz

# 2) retarget + ablations + export viewer data (main env)
python run_retarget.py --grab-pre grab_frames/cylindersmall_lift_f1034.npz \
    --contactdb ../grab_raw/contactdb_meshes --n_obj 80 \
    --var TR_DATA_GRAB --out viewer/data_grab.js
```

`grab_extract.py` forwards GRAB's right-hand `fullpose` (45-D axis-angle) through
MANO (`smplx`, `use_pca=False, flat_hand_mean=True`), maps the 16 MANO joints + 5
fingertip vertices into our 21-keypoint order, and records the object name + 6-DoF
pose. `data.py::load_grab_preprocessed` poses the ContactDB mesh (auto mm→m),
recentres the scene at the object, and returns a `Frame` — identical downstream to
the synthetic path. To pick a clean grasp, search frames by fingertip-to-surface
distance (see the search snippet in the project notes).

**Verified result** (GRAB `cylindersmall_lift` f1034): full method
**6.6 mm contact, 0 mm penetration**; `no_IM` degrades to ~13.9 mm — i.e. the
interaction term is doing real work on real data. (Penetration ablation is mild on
this gentle grasp; it's dramatic on the synthetic tight grasp.)

### Multiple showcases at once
`grab_showcase.py` (conda env) searches GRAB s1 for the cleanest grasp frame per
object and dumps a `grab_frames/<key>.npz` (21 keypoints **+ the MANO surface
mesh** + object pose) for each. Then one `run_retarget.py` per showcase writes a
`viewer/data_<key>.js` with its own JS global:

```bash
conda run -n manoconv python grab_showcase.py        # -> grab_frames/grab_*.npz
for k in grab_cylinder grab_mug grab_sphere grab_binocular grab_camera grab_apple; do
  python run_retarget.py --grab-pre grab_frames/$k.npz --contactdb ../grab_raw/contactdb_meshes \
     --n_obj 80 --var ${k^^} --out viewer/data_$k.js
done
python run_retarget.py --object cylinder --var TR_SYN_CYL --out viewer/data_syn_cyl.js
python run_retarget.py --object sphere   --var TR_SYN_SPH --out viewer/data_syn_sph.js
```

### The viewer (`viewer/index.html`) — synchronized side-by-side
- **Left pane = human (MANO hand mesh + object)**, **right pane = robot (Wuji mesh
  + object)**. A single shared camera is rendered into two scissor viewports, so
  orbit / pan / zoom (drag anywhere) keeps both panes **perfectly in sync** for a
  direct pose comparison.
- **object** switch (top): 9 real GRAB grasps — cylinder, mug, sphere, binoculars,
  camera, apple, wineglass, hammer, flute. Add more by extending `REGISTRY` + the
  `<script>` tags (synthetic still works via `run_retarget.py --object ...`).
- **ablation** switch: full / no-IM / no-pen / no-bone / no-reg, with live contact /
  penetration metrics and loss curves; scrub/play the robot optimization.
- Toggles: object / anchors / keypoints+skeleton / interaction-mesh.

Other drop-in options (same `Frame` interface): ContactPose (what the paper used),
DexYCB, OakInk.

---

## Honest caveats / scope

- **Retargeting only.** This implements the paper's optimization (Stages 1-4). The
  downstream **RL tracking controller** (PPO) is *not* included.
- **Single frame.** `retarget()` solves one frame; `E_reg` already supports
  warm-starting from a previous frame for sequences (loop frames, pass `q_ref`).
- **Mesh penetration is approximate.** For arbitrary meshes (GRAB), `phi` uses a
  nearest-surface-point + (detached) normal surrogate — differentiable and correct
  in direction, but not an exact SDF. Primitive objects use exact analytic SDFs.
- **Weights** were tuned on the synthetic grasp; GRAB objects are at a similar
  (metre) scale so they should transfer, but you may retune `DEFAULT_WEIGHTS`.
- **Keypoint correspondence** (which Wuji link = which MANO joint) is a modelling
  choice in `_keypoint_frames()`; adjust if you prefer different frames.
- **GRAB hand shape** uses the default MANO template (`betas=0`), not the
  per-subject hand mesh (`vtemp`), so keypoints are a generic-shaped hand in the
  correct pose at the correct world location — fine for retargeting, slightly off
  for exact per-subject contact.
- Contact on a larger robot hand won't be sub-mm like a same-size hand — ~3 mm with
  <1 mm penetration is the realistic embodiment-gap result.

## Files
```
toporetarget/
  geometry.py     rotations (6-D), Kabsch, object SDFs (sphere/cylinder/mesh)
  robot.py        Wuji URDF FK, 21-keypoint correspondence, limits, Procrustes init
  interaction.py  Delaunay edges, distance-aware weights, Laplacian operator
  optimize.py     loss terms + the optimization driver (ablation toggles, logging)
  data.py         Frame container, synthetic grasp generator, GRAB loader
run_retarget.py   run full+ablations, export viewer/data.js
viewer/index.html three.js viewer (orbit, scrub, ablation toggles, loss curves)
```
