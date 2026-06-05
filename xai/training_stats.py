"""
training_stats.py — Compute and cache training-set statistics for XAI baselines.

Computes the per-feature median and mean of the 10 per-flow path scalars
over the full all_multiplexed training split.

Saved to results/training_stats.json so it is computed once and reused
by both IG and KernelSHAP.

Usage:
    python xai/training_stats.py          # compute and save
    from xai.training_stats import load_training_medians
    medians = load_training_medians()     # load cached values
"""

import os
import sys
import json

import numpy as np

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELAY_DIR  = os.path.join(REPO_ROOT, 'traffic_models', 'delay')
TRAIN_PATH = os.path.join(REPO_ROOT, 'data', 'traffic_models', 'all_multiplexed', 'train')
OUT_PATH   = os.path.join(REPO_ROOT, 'results', 'training_stats.json')

sys.path.insert(0, DELAY_DIR)
sys.path.insert(0, REPO_ROOT)
os.chdir(DELAY_DIR)

# The 10 per-flow path scalar features — XAI scope (THESIS_DECISIONS §5)
PATH_SCALAR_FEATURES = [
    'traffic',
    'packets',
    'eq_lambda',
    'avg_pkts_lambda',
    'exp_max_factor',
    'pkts_lambda_on',
    'avg_t_off',
    'avg_t_on',
    'ar_a',
    'sigma',
]


def compute_training_stats(train_path=TRAIN_PATH, out_path=OUT_PATH, max_sims=500):
    """
    Iterate over the training dataset, collect all flow-level values for each
    of the 10 path scalar features, compute median and mean per feature.

    Args:
        train_path: path to all_multiplexed/train/
        out_path:   where to save the JSON result
        max_sims:   if not None, stop after this many simulations (for testing)

    Returns:
        dict with keys 'median' and 'mean', each mapping feature -> float
    """
    import tensorflow as tf
    from data_generator import input_fn

    print("Computing training-set statistics for all_multiplexed...")
    ds = input_fn(train_path, shuffle=False)

    collectors = {f: [] for f in PATH_SCALAR_FEATURES}
    n_sims = 0

    for x_batch, _ in ds:
        for feat in PATH_SCALAR_FEATURES:
            vals = x_batch[feat].numpy().flatten()
            collectors[feat].extend(vals.tolist())
        n_sims += 1
        if n_sims % 200 == 0:
            print(f"  processed {n_sims} simulations...")
        if max_sims is not None and n_sims >= max_sims:
            break

    print(f"  total simulations: {n_sims}")

    medians = {}
    means   = {}
    for feat in PATH_SCALAR_FEATURES:
        arr = np.array(collectors[feat], dtype=np.float32)
        medians[feat] = float(np.median(arr))
        means[feat]   = float(np.mean(arr))
        print(f"  {feat:20s}  median={medians[feat]:.6f}  mean={means[feat]:.6f}")

    result = {
        'n_simulations': n_sims,
        'features':      PATH_SCALAR_FEATURES,
        'median':        medians,
        'mean':          means,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved to {out_path}")
    return result


def load_training_medians(out_path=OUT_PATH):
    """
    Load cached training-set medians. Computes them if not yet cached.

    Returns:
        dict mapping feature_name -> median_value (float)
    """
    if not os.path.exists(out_path):
        print(f"[training_stats] Cache not found at {out_path}. Computing now...")
        stats = compute_training_stats()
    else:
        with open(out_path) as fh:
            stats = json.load(fh)
    return stats['median']


def load_training_means(out_path=OUT_PATH):
    """Load cached training-set means."""
    if not os.path.exists(out_path):
        stats = compute_training_stats()
    else:
        with open(out_path) as fh:
            stats = json.load(fh)
    return stats['mean']


if __name__ == '__main__':
    compute_training_stats()
