"""
Step 1 inspection script — exploring all 5 traffic_models test sets.
Mirrors predict.py imports/paths but does NOT load any model.
"""
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append('../../')
from data_generator import input_fn

for tm in ['constant_bitrate', 'onoff', 'autocorrelated', 'modulated', 'all_multiplexed']:
    TEST_PATH = f'../../data/traffic_models/{tm}/test'

    print(f"\n========== INSPECTING: {tm.upper()} ==========")
    print(f"path: {TEST_PATH}")
    print(f"exists: {os.path.isdir(TEST_PATH)}")
    if not os.path.isdir(TEST_PATH):
        print("  >>> directory missing, skipping")
        continue

    ds_test = input_fn(TEST_PATH, shuffle=False)
    ds_test = ds_test.take(200)

    # First sample: feature shapes + ranges
    for x, y in ds_test.take(1):
        print("\n--- First sample feature ranges ---")
        print("  policy unique:  ", np.unique(x["policy"].numpy()))
        print("  model unique:   ", np.unique(x["model"].numpy()))
        print("  priority unique:", np.unique(x["priority"].numpy()))
        for key in ["traffic", "packets", "eq_lambda", "capacity",
                    "queue_size", "weight", "length"]:
            arr = x[key].numpy()
            print(f"  {key:18s} min={arr.min():.4g}  max={arr.max():.4g}  mean={arr.mean():.4g}")
        yn = y.numpy()
        print(f"  LABEL              min={yn.min():.4g}  max={yn.max():.4g}  mean={yn.mean():.4g}")
        break

    # Full pass: counts + models seen
    n_samples, n_flows_total = 0, 0
    models_seen = set()
    policies_seen = set()
    for x, y in ds_test:
        n_samples += 1
        n_flows_total += int(x["traffic"].shape[0])
        models_seen.update(np.unique(x["model"].numpy()).tolist())
        policies_seen.update(np.unique(x["policy"].numpy()).tolist())
    print(f"\n--- Totals across {n_samples} samples ---")
    print(f"  total flows:    {n_flows_total}")
    print(f"  models present: {sorted(models_seen)}")
    print(f"  policies present: {sorted(policies_seen)}")

print("\n========================================\n")