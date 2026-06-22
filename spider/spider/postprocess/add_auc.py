# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ADD-AUC metric for rigid-body object tracking.

References:
  - dexmachina paper: arxiv.org/abs/2505.24853
  - Spec: ADD = mean L2 distance over sampled vertices in world frame.
    For a single rigid body sharing one transform across all vertices,
    ADD reduces to ||t_achieved - t_target||_2.
"""

import numpy as np


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration with numpy>=2.0 ``trapezoid`` and a fallback."""
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        fn = np.trapz  # noqa: NPY201 - fallback for numpy<2.0
    return float(fn(y, x))


def compute_add_auc(
    achieved_position: np.ndarray,
    target_position: np.ndarray,
    max_threshold: float = 0.1,
    num_points: int = 100,
) -> float:
    """Returns the AUC of the success-rate curve over [0, max_threshold].

    Args:
        achieved_position: (T, 3) world-frame object positions from rollout.
        target_position: (T, 3) world-frame object positions from reference.
        max_threshold: upper distance threshold (meters).
        num_points: number of threshold samples used to build the curve.

    Returns:
        AUC of the per-frame success rate vs. threshold curve, with the
        threshold axis normalized to [0, 1] so the result is in [0, 1].
    """
    assert achieved_position.shape == target_position.shape
    assert achieved_position.shape[-1] == 3
    l2 = np.linalg.norm(achieved_position - target_position, axis=-1)  # (T,)
    thresholds = np.linspace(0.0, max_threshold, num_points)
    success = (l2[:, None] <= thresholds[None, :]).astype(np.float64).mean(axis=0)
    normalized_thresholds = thresholds / max_threshold
    return _trapezoid(success, normalized_thresholds)


def compute_add_auc10(
    achieved_position: np.ndarray,
    target_position: np.ndarray,
    num_points: int = 100,
) -> float:
    """ADD-AUC integrated over thresholds in [0, 10] cm."""
    return compute_add_auc(
        achieved_position,
        target_position,
        max_threshold=0.1,
        num_points=num_points,
    )
