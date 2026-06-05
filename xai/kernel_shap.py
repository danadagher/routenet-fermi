"""
kernel_shap.py — KernelSHAP for RouteNet-Fermi.

Computes SHAP values for the 10 per-flow path scalars for a single target
flow (flow_idx=0) in each simulation.

Method: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
        NeurIPS 2017. Uses shap.KernelExplainer.

Wrapper design (consistent with CBR pilot):
  - The full graph (links, queues, all flows, topology) is fixed.
  - Only flow[flow_idx]'s 10 path scalars are perturbed.
  - Background reference: single point = training-set median per feature.
  - nsamples = 256 perturbations per simulation.

The 12 structural inputs are never perturbed (THESIS_DECISIONS §5).
"""

import os
import sys
import time

import numpy as np
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from xai.training_stats import PATH_SCALAR_FEATURES, load_training_medians

N_PERTURBATIONS = 256   # per THESIS_DECISIONS §8
FLOW_IDX        = 0     # per THESIS_DECISIONS §8


# ── KernelSHAP black-box wrapper ───────────────────────────────────────────────

class RouteNetSHAPWrapper:
    """
    Black-box wrapper around RouteNet-Fermi for shap.KernelExplainer.

    Exposes a predict(X) function where:
      X: np.ndarray of shape (n_samples, 10) — the 10 path scalar values
         for flow[flow_idx]
      returns: np.ndarray of shape (n_samples,) — model output for flow[flow_idx]

    Everything outside flow[flow_idx]'s 10 path scalars is held fixed at the
    actual simulation values.
    """

    def __init__(self, model, inputs_fixed, flow_idx=FLOW_IDX):
        """
        Args:
            model:        compiled + loaded RouteNet_Fermi
            inputs_fixed: dict of tensors for one simulation (from input_fn)
            flow_idx:     index of the target flow (default 0)
        """
        self.model      = model
        self.inputs_fix = inputs_fixed
        self.flow_idx   = flow_idx
        # Pre-cache the numpy arrays for mutable path scalar features
        self._base = {f: inputs_fixed[f].numpy().copy()
                      for f in PATH_SCALAR_FEATURES}
        self._n_flows = int(inputs_fixed['traffic'].shape[0])

    def predict(self, X):
        """
        Args:
            X: np.ndarray (n_samples, 10) — one row per perturbation,
               columns ordered as PATH_SCALAR_FEATURES

        Returns:
            np.ndarray (n_samples,) — model prediction for flow[flow_idx]
        """
        outputs = []
        for row in X:
            # Build modified input dict
            modified = dict(self.inputs_fix)
            for i, feat in enumerate(PATH_SCALAR_FEATURES):
                arr = self._base[feat].copy()          # (n_flows, 1)
                arr[self.flow_idx, 0] = float(row[i])
                modified[feat] = tf.constant(arr, dtype=tf.float32)

            pred = self.model(modified, training=False)
            outputs.append(float(pred[self.flow_idx]))

        return np.array(outputs, dtype=np.float32)


# ── single-simulation SHAP computation ────────────────────────────────────────

def compute_kernel_shap(model, inputs, medians,
                        n_perturbations=N_PERTURBATIONS,
                        flow_idx=FLOW_IDX):
    """
    Compute KernelSHAP values for flow[flow_idx] in a single simulation.

    Args:
        model:           compiled + loaded RouteNet_Fermi
        inputs:          dict of tensors for one simulation
        medians:         dict feature -> training-set median (background)
        n_perturbations: number of SHAP samples (default 256)
        flow_idx:        target flow index (default 0)

    Returns:
        shap_scores: dict feature -> float (signed SHAP value)
    """
    import shap

    wrapper = RouteNetSHAPWrapper(model, inputs, flow_idx=flow_idx)

    # Background: single reference point = training-set medians
    background = np.array([[medians[f] for f in PATH_SCALAR_FEATURES]],
                          dtype=np.float32)  # (1, 10)

    explainer = shap.KernelExplainer(wrapper.predict, background)

    # Actual feature values for the target flow
    x_explain = np.array([[float(inputs[f][flow_idx, 0])
                           for f in PATH_SCALAR_FEATURES]],
                         dtype=np.float32)  # (1, 10)

    shap_vals = explainer.shap_values(x_explain, nsamples=n_perturbations,
                                      silent=True)
    # shap_vals: (1, 10) — one row since we explain one instance
    shap_arr = np.array(shap_vals).flatten()  # (10,)

    return {feat: float(shap_arr[i]) for i, feat in enumerate(PATH_SCALAR_FEATURES)}


# ── batch over dataset ─────────────────────────────────────────────────────────

def run_kernel_shap_on_dataset(model, ds, medians,
                               n_perturbations=N_PERTURBATIONS,
                               flow_idx=FLOW_IDX):
    """
    Run KernelSHAP over all simulations in a dataset.

    Args:
        model:           compiled + loaded RouteNet_Fermi
        ds:              tf.data.Dataset of simulations
        medians:         dict from load_training_medians()
        n_perturbations: SHAP samples per simulation (default 256)
        flow_idx:        target flow index (default 0)

    Returns:
        results:  list of dicts {feature -> shap_score}, one per simulation
        timings:  list of wall-clock seconds, one per simulation
    """
    results = []
    timings = []

    for i, (x_batch, _) in enumerate(ds):
        t0 = time.time()
        scores = compute_kernel_shap(model, x_batch, medians,
                                     n_perturbations=n_perturbations,
                                     flow_idx=flow_idx)
        elapsed = time.time() - t0
        results.append(scores)
        timings.append(elapsed)
        if (i + 1) % 10 == 0:
            print(f"  KernelSHAP: {i+1} sims done  "
                  f"(last={elapsed:.1f}s, avg={np.mean(timings):.1f}s)")

    return results, timings


# ── aggregation ────────────────────────────────────────────────────────────────

def aggregate_shap_ranking(shap_results):
    """
    Aggregate per-simulation SHAP scores into a global feature ranking.

    Uses mean(|shap_score|) across all simulations.

    Args:
        shap_results: list of dicts {feature -> shap_score}

    Returns:
        ranking: list of (feature, mean_abs_score) sorted descending
    """
    sums = {f: 0.0 for f in PATH_SCALAR_FEATURES}
    n = len(shap_results)
    for scores in shap_results:
        for feat in PATH_SCALAR_FEATURES:
            sums[feat] += abs(scores[feat])
    mean_abs = {f: sums[f] / n for f in PATH_SCALAR_FEATURES}
    ranking = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)
    return ranking
