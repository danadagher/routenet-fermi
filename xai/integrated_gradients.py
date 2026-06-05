"""
integrated_gradients.py — Integrated Gradients for RouteNet-Fermi.

Computes per-feature attribution scores for the 10 per-flow path scalars
(THESIS_DECISIONS §5) for a single target flow (flow_idx=0) in each simulation.

Method: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017.

Baseline: training-set median for each of the 10 path scalar features.
          All other flows and all 12 structural inputs are unchanged.
Steps:    N_STEPS = 50 interpolation steps (trapezoid rule approximation).

The 12 structural inputs (length, model, link features, queue features,
graph tensors) are NEVER perturbed — they define the graph structure the
GNN operates over (THESIS_DECISIONS §5).
"""

import os
import sys

import numpy as np
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from xai.training_stats import PATH_SCALAR_FEATURES, load_training_medians

N_STEPS = 50   # interpolation steps (paper standard)


# ── core IG computation ────────────────────────────────────────────────────────

def _interpolate_flow0(inputs, medians, alpha):
    """
    Return a copy of `inputs` where flow[0]'s 10 path scalars are interpolated
    from their training-set median (alpha=0) to their actual value (alpha=1).
    All other flows and all structural inputs are unchanged.

    Args:
        inputs:  dict of tensors for one simulation
        medians: dict feature -> median float
        alpha:   float in [0, 1]

    Returns:
        new dict of tensors (path scalars for flow[0] interpolated)
    """
    result = {}
    for key, val in inputs.items():
        if key in PATH_SCALAR_FEATURES:
            # val: (n_flows, 1)  dtype float32
            median_val = tf.cast(medians[key], val.dtype)
            interp_row = median_val + alpha * (val[0:1] - median_val)  # (1, 1)
            if val.shape[0] is not None and val.shape[0] == 1:
                result[key] = interp_row
            else:
                result[key] = tf.concat([interp_row, val[1:]], axis=0)
        else:
            result[key] = val
    return result


def compute_ig(model, inputs, medians, n_steps=N_STEPS, flow_idx=0):
    """
    Compute Integrated Gradients for flow[flow_idx] in a single simulation.

    Only flow[flow_idx]'s path scalars are interpolated from baseline to actual.
    All other inputs (other flows + structural) remain at their actual values.

    Args:
        model:      compiled RouteNet_Fermi, weights already loaded
        inputs:     dict of tensors for one simulation (from input_fn)
        medians:    dict feature -> training-set median (from load_training_medians)
        n_steps:    number of interpolation steps (default 50)
        flow_idx:   which flow to attribute (default 0, per THESIS_DECISIONS §8)

    Returns:
        ig_scores: dict feature -> float (signed attribution for flow[flow_idx])
        The absolute values |ig_scores[f]| are used for global ranking.
    """
    alphas = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)
    accumulated_grads = {f: 0.0 for f in PATH_SCALAR_FEATURES}

    for alpha in alphas:
        interp = _interpolate_flow0(inputs, medians, float(alpha))
        watched = [interp[f] for f in PATH_SCALAR_FEATURES]

        with tf.GradientTape() as tape:
            for t in watched:
                tape.watch(t)
            output = model(interp, training=False)
            target = output[flow_idx]  # scalar prediction for this flow

        grads = tape.gradient(target, watched)
        # grads[i]: (n_flows, 1) — only row flow_idx is non-trivially affected
        # by the interpolation, but the GNN message passing can spread influence
        for i, feat in enumerate(PATH_SCALAR_FEATURES):
            if grads[i] is not None:
                accumulated_grads[feat] += float(grads[i][flow_idx, 0])

    # Trapezoid rule: divide by n_steps (n_steps+1 evaluations including endpoints)
    ig_scores = {}
    for feat in PATH_SCALAR_FEATURES:
        x_i      = float(inputs[feat][flow_idx, 0])
        x_prime  = float(medians[feat])
        avg_grad = accumulated_grads[feat] / n_steps
        ig_scores[feat] = (x_i - x_prime) * avg_grad

    return ig_scores


def run_ig_on_dataset(model, ds, medians, n_steps=N_STEPS, flow_idx=0):
    """
    Run IG over all simulations in a dataset and return per-simulation results.

    Args:
        model:    compiled + loaded RouteNet_Fermi
        ds:       tf.data.Dataset of simulations
        medians:  dict from load_training_medians()
        n_steps:  interpolation steps (default 50)
        flow_idx: target flow index (default 0)

    Returns:
        list of dicts, one per simulation, each {feature -> ig_score (float)}
    """
    results = []
    for i, (x_batch, _) in enumerate(ds):
        scores = compute_ig(model, x_batch, medians, n_steps=n_steps, flow_idx=flow_idx)
        results.append(scores)
        if (i + 1) % 50 == 0:
            print(f"  IG: {i + 1} simulations done")
    return results


def aggregate_ig_ranking(ig_results):
    """
    Aggregate per-simulation IG scores into a global feature ranking.

    Uses mean(|ig_score|) across all simulations, per PIPELINE.md Step 5.

    Args:
        ig_results: list of dicts {feature -> ig_score} (one per simulation)

    Returns:
        ranking: list of (feature, mean_abs_score) sorted descending
    """
    sums = {f: 0.0 for f in PATH_SCALAR_FEATURES}
    n = len(ig_results)
    for scores in ig_results:
        for feat in PATH_SCALAR_FEATURES:
            sums[feat] += abs(scores[feat])
    mean_abs = {f: sums[f] / n for f in PATH_SCALAR_FEATURES}
    ranking = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)
    return ranking
