"""
Step 4 — Analyze SHAP importance scores

Loads results from Step 3 and produces:
  - Feature importance table (sorted, with % contribution)
  - Per-feature direction analysis (does high value increase or decrease delay?)
  - Masking decision table (keep / mask)
  - Correlation between feature value and SHAP value (sign consistency check)
  - Additional plots: SHAP vs feature value scatter for top features
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load Step 3 outputs ────────────────────────────────────────────────────────

shap_values  = np.load('shap_values_constant_bitrate.npy')   # (40, 10)
sel_delays   = np.load('shap_delays_selected.npy')           # (40,)
sel_features = np.load('shap_features_selected.npy')         # (40, 10)
groups       = np.load('shap_groups.npy', allow_pickle=True) # (40,) ['low'/'high']

PATH_FEATURE_NAMES = [
    'traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda',
    'exp_max_factor', 'pkts_lambda_on', 'avg_t_off', 'avg_t_on',
    'ar_a', 'sigma'
]
N = len(PATH_FEATURE_NAMES)

os.makedirs('shap_plots', exist_ok=True)


# ── 1. Global importance table ─────────────────────────────────────────────────

mean_abs   = np.abs(shap_values).mean(axis=0)
total_imp  = mean_abs.sum()
pct        = mean_abs / total_imp * 100
order      = np.argsort(mean_abs)[::-1]

print("=" * 65)
print("STEP 4 — Feature Importance Analysis")
print("=" * 65)
print(f"\n{'Rank':<5} {'Feature':<22} {'Mean |SHAP|':>12} {'% Total':>9}  Decision")
print("-" * 65)
for rank, i in enumerate(order, 1):
    imp = mean_abs[i]
    # masking decision: zero or near-zero (<1% of total)
    decision = "KEEP" if pct[i] >= 1.0 else "MASK (zero out)"
    print(f"  {rank:<3} {PATH_FEATURE_NAMES[i]:<22} {imp:>12.6f} {pct[i]:>8.2f}%  {decision}")

print("-" * 65)
print(f"  {'Total':<26} {total_imp:>12.6f} {'100.00%':>9}")


# ── 2. Direction analysis (sign of SHAP vs feature value) ─────────────────────

print("\n\nDirection Analysis — how does each feature affect delay?")
print(f"{'Feature':<22}  {'Mean SHAP':>10}  {'Positive%':>10}  {'Interpretation'}")
print("-" * 75)

for i in order:
    sv_col   = shap_values[:, i]
    fv_col   = sel_features[:, i]
    mean_sv  = sv_col.mean()
    pos_pct  = (sv_col > 0).mean() * 100

    if mean_abs[i] < 1e-8:
        interp = "No effect (constant in CBR)"
    elif mean_sv > 0:
        interp = "Higher value -> MORE delay"
    else:
        interp = "Higher value -> LESS delay"

    print(f"{PATH_FEATURE_NAMES[i]:<22}  {mean_sv:>10.6f}  {pos_pct:>9.1f}%  {interp}")


# ── 3. Low vs High regime comparison ──────────────────────────────────────────

low_mask  = groups == 'low'
high_mask = groups == 'high'

low_abs  = np.abs(shap_values[low_mask]).mean(axis=0)
high_abs = np.abs(shap_values[high_mask]).mean(axis=0)

print("\n\nLow vs High Delay Regime — Importance Shift")
print(f"{'Feature':<22}  {'Low delay':>10}  {'High delay':>11}  {'Ratio H/L':>10}  {'Shift'}")
print("-" * 80)
for i in order:
    ratio = high_abs[i] / low_abs[i] if low_abs[i] > 1e-10 else float('inf')
    if ratio > 2:
        shift = "MORE important at high delay"
    elif ratio < 0.5:
        shift = "LESS important at high delay"
    elif mean_abs[i] < 1e-8:
        shift = "irrelevant both"
    else:
        shift = "stable"
    ratio_str = f"{ratio:.2f}x" if ratio != float('inf') else "inf"
    print(f"{PATH_FEATURE_NAMES[i]:<22}  {low_abs[i]:>10.6f}  {high_abs[i]:>11.6f}  "
          f"{ratio_str:>10}  {shift}")


# ── 4. Masking recommendation summary ─────────────────────────────────────────

keep_feats = [PATH_FEATURE_NAMES[i] for i in range(N) if pct[i] >= 1.0]
mask_feats = [PATH_FEATURE_NAMES[i] for i in range(N) if pct[i] < 1.0]

print("\n\n" + "=" * 65)
print("MASKING RECOMMENDATION (Step 5)")
print("=" * 65)
print(f"\nKEEP ({len(keep_feats)} features — drive >99% of importance):")
for f in keep_feats:
    print(f"  + {f}")

print(f"\nMASK / zero out ({len(mask_feats)} features — <1% each, irrelevant for CBR):")
for f in mask_feats:
    print(f"  - {f}")

print(f"\nInput reduction: {N} -> {len(keep_feats)} path features  "
      f"({100*(N-len(keep_feats))/N:.0f}% reduction)")


# ── 5. Scatter plots: SHAP value vs feature value (top 4 features) ────────────

top4 = order[:4]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, i in zip(axes, top4):
    fv = sel_features[:, i]
    sv = shap_values[:, i]
    colors = ['tomato' if g == 'high' else 'cornflowerblue' for g in groups]
    ax.scatter(fv, sv, c=colors, s=60, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel(PATH_FEATURE_NAMES[i], fontsize=10)
    ax.set_ylabel('SHAP value (seconds)', fontsize=10)
    ax.set_title(f'{PATH_FEATURE_NAMES[i]}  |  imp={mean_abs[i]:.4f}s  ({pct[i]:.1f}%)',
                 fontsize=10)
    # legend proxy
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='tomato', label='High delay'),
                       Patch(facecolor='cornflowerblue', label='Low delay')]
    ax.legend(handles=legend_elements, fontsize=8)

plt.suptitle('SHAP Value vs Feature Value — Top 4 Features\n'
             'constant_bitrate | RouteNet-Fermi Delay Model', fontsize=11)
plt.tight_layout()
plt.savefig('shap_plots/shap_vs_feature_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: shap_plots/shap_vs_feature_scatter.png")


# ── 6. Cumulative importance curve ────────────────────────────────────────────

cumulative = np.cumsum(mean_abs[order]) / total_imp * 100

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(N), pct[order], color='steelblue', alpha=0.7, label='Individual %')
ax.plot(range(N), cumulative, 'o-', color='tomato', linewidth=2, label='Cumulative %')
ax.axhline(99, color='gray', linestyle='--', linewidth=0.8, label='99% threshold')
ax.set_xticks(range(N))
ax.set_xticklabels([PATH_FEATURE_NAMES[i] for i in order], rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Importance (%)', fontsize=11)
ax.set_title('Cumulative Feature Importance — constant_bitrate\nKernelSHAP', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig('shap_plots/cumulative_importance.png', dpi=150)
plt.close()
print("Saved: shap_plots/cumulative_importance.png")

print("\nStep 4 complete.")
