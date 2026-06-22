# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
    Retargeting quality gate: is the SAMPLE and the TRAINING good enough?

    It measures four things and turns them into an actionable verdict that distinguishes
    "need more DATA" from "need more TRAINING":

      (1) Neural-FK accuracy   [training]  -- the surrogate the IK was trained against.
      (2) Coverage             [data]      -- do deployment fingertips fall inside the
                                              trained workspace? (NN distance to train cloud)
      (3) Direction consistency[training]  -- moving a human fingertip moves the robot
                                              fingertip the same way (measured in-distribution).
      (4) Pinch fidelity       [training]  -- when human tips touch, do robot tips converge?
    Plus validity/joint-saturation sanity checks.

    Decision matrix:
      coverage OK + training OK  -> SATISFIED
      coverage BAD + training OK -> COLLECT MORE SAMPLES (which fingers is reported)
      coverage OK + training BAD -> TRAIN MORE / TUNE
      both BAD                    -> FIX DATA FIRST, then retrain

    Usage (geort env):
      python geort/mocap/eval_retarget.py -hand wuji -ckpt_tag wuji_mano \
          --train_data mano_right --eval_data furelise_65,grab_s2_stapler
'''

import argparse
import contextlib
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

from geort import load_model, get_config
from geort.utils.path import get_data_root, get_checkpoint_root
from geort.utils.config_utils import parse_config_keypoint_info
from geort.env.hand import HandKinematicModel
from geort.formatter import HandFormatter
from geort.model import FKModel

FINGERS_DEFAULT = ["thumb", "index", "middle", "ring", "pinky"]


def tag(value, good, warn, higher_is_better=False):
    '''Return ('PASS'/'WARN'/'FAIL') for a metric given thresholds.'''
    if higher_is_better:
        return "PASS" if value >= good else ("WARN" if value >= warn else "FAIL")
    return "PASS" if value <= good else ("WARN" if value <= warn else "FAIL")


def load_npy(name):
    p = os.path.join(get_data_root(), f"{name}.npy")
    return np.load(p) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-hand", default="wuji")
    ap.add_argument("-ckpt_tag", default="wuji_mano")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--train_data", default="mano_right", help="data the model was trained on")
    ap.add_argument("--eval_data", default="", help="comma-separated deployment datasets to test coverage on")
    # thresholds (good / warn boundaries)
    ap.add_argument("--fk_good_mm", type=float, default=5.0)
    ap.add_argument("--fk_warn_mm", type=float, default=10.0)
    ap.add_argument("--cov_good_cm", type=float, default=2.0)   # 95th-pct NN distance
    ap.add_argument("--cov_warn_cm", type=float, default=4.0)
    ap.add_argument("--dir_good", type=float, default=0.90)     # cosine
    ap.add_argument("--dir_warn", type=float, default=0.80)
    ap.add_argument("--pinch_good_cm", type=float, default=2.5)
    ap.add_argument("--pinch_warn_cm", type=float, default=4.0)
    ap.add_argument("--n_dir", type=int, default=300)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    cfg = get_config(args.hand)
    info = parse_config_keypoint_info(cfg)
    human_ids = info["human_id"]
    fingers = [f.get("name", FINGERS_DEFAULT[i]) for i, f in enumerate(cfg["fingertip_link"])]
    n_f = len(fingers)

    model = load_model(args.ckpt_tag, epoch=args.epoch)
    with contextlib.redirect_stdout(open(os.devnull, "w")):  # silence SAPIEN build spam
        hand = HandKinematicModel.build_from_config(cfg, render=False)
        hand.initialize_keypoint(keypoint_link_names=[f["link"] for f in cfg["fingertip_link"]],
                                 keypoint_offsets=[f["center_offset"] for f in cfg["fingertip_link"]])
    lo, hi = [np.array(x) for x in hand.get_joint_limit()]

    def robot_tips(q):
        return hand.keypoint_from_qpos(q, ret_vec=True)  # [n_f,3] config order, base frame

    train = load_npy(args.train_data)
    if train is None:
        raise SystemExit(f"train_data '{args.train_data}' not found in data/")
    evals = [e for e in args.eval_data.split(",") if e]

    results = {}  # metric -> (value, status)
    print("=" * 70)
    print(f"GeoRT retargeting eval  | hand={args.hand}  ckpt={args.ckpt_tag}")
    print("=" * 70)

    # ---------------- (1) Neural-FK accuracy (foundational, TRAINING axis) ----------------
    fk_status = "SKIP"; fk_rms = None
    fk_ckpt = os.path.join(get_checkpoint_root(), f"fk_model_{cfg['name']}.pth")
    if os.path.exists(fk_ckpt):
        fkm = FKModel(keypoint_joints=info["joint"]).cuda()
        fkm.load_state_dict(torch.load(fk_ckpt)); fkm.eval()
        fmt = HandFormatter(lo, hi)
        qs = rng.uniform(0, 1, (2000, len(lo))) * (hi - lo) + lo
        true = np.array([robot_tips(q) for q in qs])              # [N,n_f,3] true SAPIEN FK
        with torch.no_grad():
            pred = fkm(fmt.normalize_torch(torch.from_numpy(qs).cuda().float())).cpu().numpy()
        fk_rms = float(np.sqrt(((pred - true) ** 2).sum(-1)).mean()) * 1000  # mm
        fk_status = tag(fk_rms, args.fk_good_mm, args.fk_warn_mm)
        print(f"\n[1] Neural-FK accuracy (training): RMS {fk_rms:.2f} mm   [{fk_status}]")
    else:
        print(f"\n[1] Neural-FK accuracy: SKIP (no {fk_ckpt})")
    results["fk"] = (fk_rms, fk_status)

    # ---------------- (3) Direction consistency (TRAINING axis, in-distribution) ----------
    eps = 0.01
    cos = {f: [] for f in fingers}
    idx = rng.integers(0, len(train), args.n_dir)
    for t in idx:
        kp = train[t]; r1 = robot_tips(model.forward(kp))
        for fi, hid in enumerate(human_ids):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            kp2 = kp.copy(); kp2[hid] = kp2[hid] + eps * d
            rd = robot_tips(model.forward(kp2))[fi] - r1[fi]
            if np.linalg.norm(rd) > 1e-6:
                cos[fingers[fi]].append(float(d @ (rd / np.linalg.norm(rd))))
    dir_overall = float(np.mean([c for f in fingers for c in cos[f]]))
    dir_status = tag(dir_overall, args.dir_good, args.dir_warn, higher_is_better=True)
    print(f"\n[2] Direction consistency (training): overall cos {dir_overall:.3f}   [{dir_status}]")
    for f in fingers:
        print(f"      {f:8s} {np.mean(cos[f]):.3f}")
    results["dir"] = (dir_overall, dir_status)

    # ---------------- validity / joint saturation (TRAINING health) ----------------------
    sample = train[rng.integers(0, len(train), min(1000, len(train)))]
    Q = np.array([model.forward(f) for f in sample])
    finite = bool(np.isfinite(Q).all())
    span = (hi - lo)
    at_limit = float(np.mean((Q < lo + 0.02 * span) | (Q > hi - 0.02 * span)))
    print(f"\n[3] Validity: finite={finite} | mean fraction of joints at limit = {at_limit*100:.1f}%")
    results["finite"] = finite

    # ---------------- (2) Coverage (DATA axis) + (4) Pinch fidelity (per eval set) --------
    trees = [cKDTree(train[:, hid, :]) for hid in human_ids]
    cov_status_all, pinch_status_all = [], []
    cov_bad_fingers = {}
    if not evals:
        print("\n[4] Coverage/pinch: SKIP (no --eval_data given; cannot assess deployment coverage)")
    for name in evals:
        kp = load_npy(name)
        if kp is None:
            print(f"\n[4] {name}: NOT FOUND, skipped"); continue
        sub = kp[:: max(1, len(kp) // 2000)]
        # coverage per finger
        p95s = []
        bad = []
        for fi, hid in enumerate(human_ids):
            dd = trees[fi].query(sub[:, hid, :])[0] * 100  # cm
            p95 = np.percentile(dd, 95); p95s.append(p95)
            if p95 > args.cov_warn_cm:
                bad.append(f"{fingers[fi]}({p95:.1f}cm)")
        cov_p95 = max(p95s)
        cstat = tag(cov_p95, args.cov_good_cm, args.cov_warn_cm)
        cov_status_all.append(cstat)
        if bad:
            cov_bad_fingers[name] = bad
        # pinch fidelity (thumb vs other fingers)
        Rt = np.array([robot_tips(model.forward(f)) for f in sub[:: max(1, len(sub) // 400)]])
        subp = sub[:: max(1, len(sub) // 400)]
        pinch_vals = []
        for fj in range(1, n_f):
            hd = np.linalg.norm(subp[:, human_ids[0]] - subp[:, human_ids[fj]], axis=1)
            close = hd < 0.015
            if close.sum() >= 3:
                pinch_vals.append(np.linalg.norm(Rt[close, 0] - Rt[close, fj], axis=1).mean() * 100)
        pinch = float(np.mean(pinch_vals)) if pinch_vals else None
        pstat = tag(pinch, args.pinch_good_cm, args.pinch_warn_cm) if pinch is not None else "N/A"
        if pinch is not None:
            pinch_status_all.append(pstat)
        print(f"\n[4] {name}: coverage p95(max finger) {cov_p95:.2f} cm [{cstat}]"
              f" | pinch gap {('%.1f cm [%s]' % (pinch, pstat)) if pinch is not None else 'no pinch frames'}")
        if bad:
            print(f"      under-covered fingers: {', '.join(bad)}")

    # ---------------- VERDICT ----------------
    def worst(statuses):
        order = {"FAIL": 3, "WARN": 2, "PASS": 1, "SKIP": 0, "N/A": 0}
        return max(statuses, key=lambda s: order.get(s, 0)) if statuses else "SKIP"

    coverage_state = worst(cov_status_all) if cov_status_all else "SKIP"
    training_states = [s for s in [fk_status, dir_status] + pinch_status_all if s not in ("SKIP", "N/A")]
    training_state = worst(training_states) if training_states else "SKIP"
    data_bad = coverage_state == "FAIL"
    train_bad = (training_state == "FAIL") or (not finite)

    print("\n" + "=" * 70)
    print(f"VERDICT   data/coverage: {coverage_state}    training/quality: {training_state}")
    print("=" * 70)
    if not finite:
        print("BROKEN: model outputs non-finite qpos. Retrain from scratch.")
    elif data_bad and train_bad:
        print(">> FIX DATA FIRST, THEN RETRAIN.")
        print("   Coverage is insufficient AND the map is poor on covered regions.")
        _data_guidance(cov_bad_fingers, args)
        _train_guidance(fk_status, dir_status, pinch_status_all, args)
    elif data_bad:
        print(">> COLLECT MORE SAMPLES (the model is fine; the data doesn't cover deployment).")
        _data_guidance(cov_bad_fingers, args)
    elif train_bad or training_state == "WARN":
        print(">> TRAIN MORE / TUNE (the data covers deployment; the map underfits).")
        _train_guidance(fk_status, dir_status, pinch_status_all, args)
    elif coverage_state in ("WARN",) or training_state == "WARN":
        print(">> ACCEPTABLE but borderline — see WARN metrics above; consider a margin of more data/epochs.")
    else:
        print(">> SATISFIED. Sample coverage and training quality both pass.")
        if coverage_state == "SKIP":
            print("   (Note: no --eval_data given, so deployment coverage was NOT checked — "
                  "pass your real datasets to confirm sufficiency.)")


def _data_guidance(cov_bad_fingers, args):
    print("   ACTION (more data):")
    if cov_bad_fingers:
        for name, fl in cov_bad_fingers.items():
            print(f"     - '{name}' is out-of-distribution on: {', '.join(fl)}")
    print("     - Re-sample with wider coverage:  python geort/mocap/mano_mocap.py "
          "--name <new> --n_samples 20000 --pca_std 3.0")
    print("     - Then retrain on the combined/new data.")


def _train_guidance(fk_status, dir_status, pinch_status, args):
    print("   ACTION (more training):")
    if fk_status == "FAIL":
        print("     - Neural FK is the bottleneck: delete checkpoint/fk_model_<hand>.pth and "
              "retrain (it will rebuild the FK surrogate).")
    if dir_status in ("FAIL", "WARN"):
        print("     - Direction map underfits: train more epochs "
              "(python geort/trainer.py ... ) or verify FK first.")
    if any(s in ("FAIL", "WARN") for s in pinch_status):
        print("     - Pinch weak: raise --w_pinch (e.g. 3.0) and/or add thumb-opposition poses.")


if __name__ == "__main__":
    main()
