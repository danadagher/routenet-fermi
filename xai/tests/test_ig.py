"""
test_ig.py — Sanity checks for Integrated Gradients.

Two checks:
  1. Real-data check: load one all_multiplexed simulation, run IG with n_steps=10
     (fast), verify scores are finite and at least one is non-zero.
  2. Synthetic check: construct a fake input where one feature (traffic) is set
     to 50x its median while all others are at median. Verify that IG ranks
     'traffic' as the most important feature (highest |score|).

Run from repo root:
    python -m xai.tests.test_ig
"""

import os
import sys
import re

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DELAY_DIR  = os.path.join(REPO_ROOT, 'traffic_models', 'delay')
sys.path.insert(0, DELAY_DIR)
sys.path.insert(0, REPO_ROOT)
os.chdir(DELAY_DIR)

import numpy as np
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)

from data_generator import input_fn
from delay_model import RouteNet_Fermi
from xai.training_stats import PATH_SCALAR_FEATURES, load_training_medians
from xai.integrated_gradients import compute_ig, N_STEPS


def load_model():
    ckpt_dir = os.path.join(DELAY_DIR, 'ckpt_dir_all_multiplexed')
    best = None; best_mre = float('inf')
    for f in os.listdir(ckpt_dir):
        if os.path.isfile(os.path.join(ckpt_dir, f)):
            reg = re.findall(r'\d+\.\d+', f)
            if reg:
                mre = float(reg[0])
                if mre <= best_mre:
                    best = (f.replace('.index', '').replace('.data', '')
                             .replace('-00000-of-00001', ''))
                    best_mre = mre
    model = RouteNet_Fermi()
    model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  optimizer=tf.keras.optimizers.Adam(0.001))
    model.load_weights(os.path.join(ckpt_dir, best))
    return model


def test_real_data():
    print("\n[Test 1] Real-data IG check (n_steps=10, 1 simulation)...")
    model   = load_model()
    medians = load_training_medians()

    TEST_PATH = os.path.join(REPO_ROOT, 'data', 'traffic_models',
                             'all_multiplexed', 'test')
    ds = input_fn(TEST_PATH, shuffle=False).take(1)
    x, _ = next(iter(ds))

    scores = compute_ig(model, x, medians, n_steps=10, flow_idx=0)

    print("  IG scores:")
    for feat, val in scores.items():
        print(f"    {feat:20s}: {val:.6f}")

    # Check 1: all finite
    for feat, val in scores.items():
        assert np.isfinite(val), f"IG score for {feat} is not finite: {val}"

    # Check 2: at least one non-zero
    assert any(abs(v) > 1e-12 for v in scores.values()), \
        "All IG scores are zero — gradient computation failed"

    print("  PASSED: all scores finite, at least one non-zero.")
    return scores


def test_synthetic():
    print("\n[Test 2] Synthetic IG check — 'traffic' set to 50x median...")
    model   = load_model()
    medians = load_training_medians()

    # Load one real simulation to get a valid graph structure
    TEST_PATH = os.path.join(REPO_ROOT, 'data', 'traffic_models',
                             'all_multiplexed', 'test')
    ds = input_fn(TEST_PATH, shuffle=False).take(1)
    x, _ = next(iter(ds))

    # Build a synthetic input: all path scalars of flow[0] at median,
    # except 'traffic' which is set to 50x its median
    synthetic = dict(x)
    for feat in PATH_SCALAR_FEATURES:
        arr = x[feat].numpy().copy()
        arr[0, 0] = float(medians[feat])
        synthetic[feat] = tf.constant(arr, dtype=tf.float32)

    # Set traffic of flow[0] to 50x median
    arr = synthetic['traffic'].numpy().copy()
    arr[0, 0] = float(medians['traffic']) * 50.0
    synthetic['traffic'] = tf.constant(arr, dtype=tf.float32)

    scores = compute_ig(model, synthetic, medians, n_steps=10, flow_idx=0)

    print("  IG scores on synthetic input (traffic=50x median):")
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    for rank, (feat, val) in enumerate(ranked, 1):
        marker = " <-- should be #1" if feat == 'traffic' else ""
        print(f"    {rank}. {feat:20s}: |score|={abs(val):.6f}{marker}")

    top_feature = ranked[0][0]
    assert top_feature == 'traffic', \
        f"Expected 'traffic' to rank #1 on synthetic input, got '{top_feature}'"

    print("  PASSED: 'traffic' ranked #1 on synthetic input.")
    return scores


if __name__ == '__main__':
    scores_real      = test_real_data()
    scores_synthetic = test_synthetic()
    print("\n=== All IG tests PASSED ===")
