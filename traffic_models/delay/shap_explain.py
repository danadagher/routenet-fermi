"""
Step 3 — KernelSHAP Explainability for RouteNet-Fermi (delay model)

Dataset : constant_bitrate  (cleanest single-model baseline)
Target  : per-flow path features (10 scalars)
Method  : shap.KernelExplainer  — model-agnostic, perturbation-based
Scope   : flow index 0 of each sample (representative single flow)
Samples : 40 flows  (20 lowest-delay + 20 highest-delay)
Output  : .npy arrays + 3 plots in shap_plots/
"""

import os
import re
import sys
import numpy as np
import tensorflow as tf
import shap
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe on Windows)
import matplotlib.pyplot as plt

sys.path.append('../../')
from delay_model import RouteNet_Fermi

# ── constants ─────────────────────────────────────────────────────────────────

TM = 'constant_bitrate'

PATH_FEATURE_NAMES = [
    'traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
    'exp_max_factor', 'pkts_lambda_on', 'avg_t_off', 'avg_t_on',
    'ar_a', 'sigma'
]

N_SAMPLES   = 200    # test samples to load
N_LOW       = 20     # low-delay flows to explain
N_HIGH      = 20     # high-delay flows to explain
SHAP_N      = 256    # SHAP perturbations per flow  (↑ = more accurate, ↓ = faster)


# ── 1. Load pretrained model ──────────────────────────────────────────────────

CKPT_DIR = f'./ckpt_dir_{TM}'

model = RouteNet_Fermi()
model.compile(
    loss=tf.keras.losses.MeanAbsolutePercentageError(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    run_eagerly=False
)

best, best_mre = None, float('inf')
for f in os.listdir(CKPT_DIR):
    if os.path.isfile(os.path.join(CKPT_DIR, f)):
        reg = re.findall(r"\d+\.\d+", f)
        if reg:
            mre = float(reg[0])
            if mre <= best_mre:
                best = (f.replace('.index', '')
                          .replace('.data', '')
                          .replace('-00000-of-00001', ''))
                best_mre = mre

print(f"Loading checkpoint: {best}  (val MAPE = {best_mre:.2f}%)")
model.load_weights(os.path.join(CKPT_DIR, best))


# ── 2. Load test samples ──────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from data_generator import input_fn

TEST_PATH = f'../../data/traffic_models/{TM}/test'
ds = input_fn(TEST_PATH, shuffle=False).take(N_SAMPLES)

samples = []
for x, y in ds:
    samples.append((x, y.numpy()))

print(f"Loaded {len(samples)} samples")


# ── 3. Feature extraction helper ─────────────────────────────────────────────

def get_flow_features(inputs, flow_idx=0):
    """Return the 10 scalar path features for flow_idx as a 1-D float32 array."""
    return np.array([
        inputs['traffic'].numpy()[flow_idx, 0],
        inputs['packets'].numpy()[flow_idx, 0],
        inputs['eq_lambda'].numpy()[flow_idx, 0],
        inputs['avg_pkts_lambda'].numpy()[flow_idx, 0],
        inputs['exp_max_factor'].numpy()[flow_idx, 0],
        inputs['pkts_lambda_on'].numpy()[flow_idx, 0],
        inputs['avg_t_off'].numpy()[flow_idx, 0],
        inputs['avg_t_on'].numpy()[flow_idx, 0],
        inputs['ar_a'].numpy()[flow_idx, 0],
        inputs['sigma'].numpy()[flow_idx, 0],
    ], dtype=np.float32)


# ── 4. SHAP wrapper ───────────────────────────────────────────────────────────

def make_predict_fn(base_inputs, flow_idx=0):
    """
    Closure around a fixed graph sample.
    Returns predict_fn(perturbed [N, 10]) -> predictions [N].
    Each row in `perturbed` replaces flow_idx's scalar path features;
    the graph structure and all other flows stay unchanged.
    """
    # snapshot mutable path feature arrays once
    snap = {k: base_inputs[k].numpy().copy()
            for k in PATH_FEATURE_NAMES}

    def predict_fn(perturbed):
        out = []
        for row in perturbed:
            inp = dict(base_inputs)   # keep ragged tensors by reference (read-only)
            for j, feat in enumerate(PATH_FEATURE_NAMES):
                arr = snap[feat].copy()
                arr[flow_idx, 0] = row[j]
                inp[feat] = tf.constant(arr, dtype=tf.float32)

            pred = model(inp, training=False).numpy()
            out.append(float(pred[flow_idx]))
        return np.array(out, dtype=np.float64)

    return predict_fn


# ── 5. Select 20 low + 20 high delay flows ────────────────────────────────────

print("Running baseline predictions to select flows …")
flow0_delays = np.array([
    float(model(x, training=False).numpy()[0])
    for x, _ in samples
])

print(f"Delay stats: min={flow0_delays.min():.6f}  "
      f"median={np.median(flow0_delays):.6f}  "
      f"max={flow0_delays.max():.6f}")

order      = np.argsort(flow0_delays)
low_idx    = order[:N_LOW].tolist()
high_idx   = order[-N_HIGH:].tolist()
sel_idx    = low_idx + high_idx
group_tags = ['low'] * N_LOW + ['high'] * N_HIGH

print(f"Low-delay  range: {flow0_delays[low_idx].min():.6f} – {flow0_delays[low_idx].max():.6f}")
print(f"High-delay range: {flow0_delays[high_idx].min():.6f} – {flow0_delays[high_idx].max():.6f}")


# ── 6. Background (median reference point) ───────────────────────────────────

all_feats  = np.stack([get_flow_features(x) for x, _ in samples])   # (N, 10)
background = np.median(all_feats, axis=0, keepdims=True)             # (1, 10)

print("\nBackground (median per feature):")
for name, val in zip(PATH_FEATURE_NAMES, background[0]):
    print(f"  {name:20s} {val:.4f}")


# ── 7. KernelSHAP loop ────────────────────────────────────────────────────────

print(f"\nRunning KernelSHAP on {len(sel_idx)} flows  (nsamples={SHAP_N}) …")
shap_values_all = []

for i, idx in enumerate(sel_idx):
    x, _ = samples[idx]
    feat  = get_flow_features(x).reshape(1, -1)    # (1, 10)
    pred_fn = make_predict_fn(x, flow_idx=0)

    explainer = shap.KernelExplainer(pred_fn, background)
    sv = explainer.shap_values(feat, nsamples=SHAP_N, silent=True)
    shap_values_all.append(sv[0])                  # (10,)

    print(f"  [{i+1:2d}/{len(sel_idx)}]  sample={idx:3d}  "
          f"group={group_tags[i]:4s}  delay={flow0_delays[idx]:.6f}  "
          f"top_feat={PATH_FEATURE_NAMES[np.argmax(np.abs(sv[0]))]}")

shap_values_all  = np.array(shap_values_all)                          # (40, 10)
sel_delays       = flow0_delays[sel_idx]
sel_features     = np.stack([get_flow_features(samples[i][0]) for i in sel_idx])  # (40, 10)


# ── 8. Save raw results ───────────────────────────────────────────────────────

np.save('shap_values_constant_bitrate.npy',   shap_values_all)
np.save('shap_delays_selected.npy',           sel_delays)
np.save('shap_features_selected.npy',         sel_features)
np.save('shap_groups.npy',                    np.array(group_tags))

print("\nSaved: shap_values_constant_bitrate.npy  shap_delays_selected.npy  "
      "shap_features_selected.npy  shap_groups.npy")


# ── 9. Plots ──────────────────────────────────────────────────────────────────

os.makedirs('shap_plots', exist_ok=True)


# 9a. Global mean |SHAP| bar chart  (all 40 flows)
mean_abs = np.abs(shap_values_all).mean(axis=0)
order_feat = np.argsort(mean_abs)[::-1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(
    [PATH_FEATURE_NAMES[o] for o in order_feat],
    mean_abs[order_feat],
    color='steelblue'
)
ax.set_xlabel('Mean |SHAP value|  (seconds of delay)', fontsize=11)
ax.set_title(
    f'Feature Importance — RouteNet-Fermi Delay\n'
    f'Traffic model: {TM}  |  40 flows  |  KernelSHAP',
    fontsize=11
)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('shap_plots/mean_shap_bar.png', dpi=150)
plt.close()
print("Saved: shap_plots/mean_shap_bar.png")


# 9b. Low vs High delay — grouped bar chart
low_abs  = np.abs(shap_values_all[:N_LOW]).mean(axis=0)
high_abs = np.abs(shap_values_all[N_LOW:]).mean(axis=0)

x_pos = np.arange(len(PATH_FEATURE_NAMES))
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x_pos - 0.2, low_abs,  0.4, label=f'Low delay  (n={N_LOW})',  color='cornflowerblue')
ax.bar(x_pos + 0.2, high_abs, 0.4, label=f'High delay (n={N_HIGH})', color='tomato')
ax.set_xticks(x_pos)
ax.set_xticklabels(PATH_FEATURE_NAMES, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Mean |SHAP value|  (seconds)', fontsize=11)
ax.set_title(
    f'Low vs High Delay — Feature Importance\n'
    f'Traffic model: {TM}  |  KernelSHAP',
    fontsize=11
)
ax.legend()
plt.tight_layout()
plt.savefig('shap_plots/low_vs_high_shap.png', dpi=150)
plt.close()
print("Saved: shap_plots/low_vs_high_shap.png")


# 9c. SHAP beeswarm / summary dot plot
fig, ax = plt.subplots(figsize=(8, 6))
shap.summary_plot(
    shap_values_all,
    sel_features,
    feature_names=PATH_FEATURE_NAMES,
    show=False,
    plot_size=None
)
plt.tight_layout()
plt.savefig('shap_plots/shap_summary_dot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_plots/shap_summary_dot.png")


# 9d. Waterfall for the highest-delay flow (single flow deep-dive)
worst_local = N_LOW + np.argmax(sel_delays[N_LOW:])   # index within sel_idx
ev = float(explainer.expected_value)

fig, ax = plt.subplots(figsize=(8, 5))
sorted_i = np.argsort(shap_values_all[worst_local])
colors = ['tomato' if v > 0 else 'cornflowerblue'
          for v in shap_values_all[worst_local][sorted_i]]
ax.barh(
    [PATH_FEATURE_NAMES[i] for i in sorted_i],
    shap_values_all[worst_local][sorted_i],
    color=colors
)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('SHAP value  (seconds)', fontsize=11)
ax.set_title(
    f'SHAP Waterfall — Highest-delay flow\n'
    f'delay={sel_delays[worst_local]:.6f} s  |  base={ev:.6f} s',
    fontsize=11
)
plt.tight_layout()
plt.savefig('shap_plots/shap_waterfall_worst.png', dpi=150)
plt.close()
print("Saved: shap_plots/shap_waterfall_worst.png")


print("\nStep 3 complete -- all SHAP results saved to traffic_models/delay/shap_plots/")
