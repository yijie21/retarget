# Visualization of SPIDER MJWP retargeting intermediate results.
# Produces PNGs explaining what the method did on arcticv2 s01-box_use_01 (xhand).
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio

OUT = "example_datasets/processed/arcticv2/xhand/bimanual/s01-box_use_01/0"
VIZ = "viz_output"
os.makedirs(VIZ, exist_ok=True)

mj = np.load(f"{OUT}/trajectory_mjwp_fast.npz", allow_pickle=True)
kin = np.load(f"{OUT}/trajectory_kinematic.npz", allow_pickle=True)

NQ = 50
NU = 36  # robot DOFs; object dims are the last 14
# bimanual object layout: [right_pos(3) right_quat(4) left_pos(3) left_quat(4)]
R_POS = slice(NU, NU + 3)
L_POS = slice(NU + 7, NU + 10)

# ---- executed trajectory (concat the MPC control ticks) ----
exq = mj["qpos"].reshape(-1, NQ)        # (300, 50)
ext = mj["time"].reshape(-1)            # (300,)
# ---- reference (human-derived kinematic) ----
refq = kin["qpos"]                      # (117, 50)
freq = float(kin["frequency"])
reft = np.arange(len(refq)) / freq

# ============ FIG 1: object position tracking (the money plot) ============
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
labels = ["X", "Y", "Z"]
for col, axis in enumerate(labels):
    # right object
    ax = axes[0, col]
    ax.plot(reft, refq[:, R_POS][:, col], "k--", lw=2, label="human reference")
    ax.plot(ext, exq[:, R_POS][:, col], "C3", lw=1.8, label="physics (SPIDER)")
    ax.set_title(f"RIGHT object  {axis}")
    if col == 0:
        ax.set_ylabel("position (m)")
    ax.grid(alpha=0.3)
    # left object
    ax = axes[1, col]
    ax.plot(reft, refq[:, L_POS][:, col], "k--", lw=2, label="human reference")
    ax.plot(ext, exq[:, L_POS][:, col], "C0", lw=1.8, label="physics (SPIDER)")
    ax.set_title(f"LEFT object  {axis}")
    ax.set_xlabel("time (s)")
    if col == 0:
        ax.set_ylabel("position (m)")
    ax.grid(alpha=0.3)
axes[0, 2].legend(loc="best", fontsize=9)
fig.suptitle("Object pose tracking: physics-retargeted robot vs. human reference\n"
             "(arcticv2 s01-box_use_01, bimanual xhand)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{VIZ}/1_object_tracking.png", dpi=110)
plt.close(fig)

# ============ FIG 2: sampling optimization per control tick ============
opt_steps = mj["opt_steps"].reshape(-1)         # (2,) actual iters per tick
rew_mean = mj["rew_mean"]; rew_max = mj["rew_max"]; rew_min = mj["rew_min"]
qdist = mj["qpos_dist_mean"]                     # tracking distance per iter
n_ticks = rew_mean.shape[0]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for t in range(n_ticks):
    n = int(opt_steps[t])
    it = np.arange(1, n + 1)
    axes[0].plot(it, rew_mean[t, :n], "-o", label=f"tick {t} mean")
    axes[0].fill_between(it, rew_min[t, :n], rew_max[t, :n], alpha=0.15)
    axes[1].plot(it, qdist[t, :n], "-o", label=f"tick {t}")
axes[0].set_title("Reward across sampling iterations\n(higher = better; band = min..max over samples)")
axes[0].set_xlabel("optimization iteration"); axes[0].set_ylabel("reward"); axes[0].grid(alpha=0.3); axes[0].legend()
axes[1].set_title("Weighted tracking distance ||x - x_ref||_Q\n(lower = closer to reference)")
axes[1].set_xlabel("optimization iteration"); axes[1].set_ylabel("distance"); axes[1].grid(alpha=0.3); axes[1].legend()
fig.suptitle("Sampling-based MPC refining each control step (Eq. 2-3)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{VIZ}/2_sampling_optimization.png", dpi=110)
plt.close(fig)

# ============ FIG 3: frame montage of physics result [ref | sim] ============
def montage(mp4, path, title, n=5):
    v = iio.imread(mp4)
    idx = np.linspace(0, len(v) - 1, n).astype(int)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.0 * n))
    if n == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        ax.imshow(v[i]); ax.axis("off")
        ax.set_title(f"frame {i}/{len(v)-1}  (t≈{i/50:.1f}s)", fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path, dpi=95); plt.close(fig)

montage(f"{OUT}/visualization_mjwp_fast.mp4", f"{VIZ}/3_physics_frames.png",
        "Physics retargeting result — LEFT half = human reference, RIGHT half = SPIDER robot (with contact points)")
montage(f"{OUT}/visualization_ik.mp4", f"{VIZ}/4_ik_frames.png",
        "Stage before physics: kinematic IK retargeting (human motion -> robot pose, no physics)")

# ---- print a short summary ----
print("nq", NQ, "ticks", n_ticks, "opt_steps", opt_steps.tolist())
print("succeeded", mj["succeeded"].reshape(-1).tolist())
print("step_pos_error(m)", np.round(mj["step_pos_error"].reshape(-1), 4).tolist())
print("step_rot_error(rad)", np.round(mj["step_rot_error"].reshape(-1), 4).tolist())
print("SAVED:", os.listdir(VIZ))
