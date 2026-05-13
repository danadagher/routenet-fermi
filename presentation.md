# XAI for GNN-Based Network Performance Prediction
## Progress Report — TRAVEL Project / WP3 Task 3.2
**Meeting:** IMT Atlantique + SogetiLabs
**Intern:** Dana Dagher (Sogeti)

---

## Slide 1 — Project Context: TRAVEL & Task 3.2

**TRAVEL** = Trustworthy and Reliable Artificial Intelligence for Vehicular Networks
→ European research project developing reliable, explainable, and trustworthy AI for V2X networks.

**Our work sits in WP3 — XAI-based End-to-End Network Automation**

> Task 3.2: *"Critical and comparative study of XAI approaches for network automation (SogetiLabs, IMT-NE). XAI schemes will be adapted to RL and GNN networks by assigning a relevance score not only to input features but also to actions taken by the RL agent."*

**IMT Contribution Plan (3 contributions):**
| Contribution | Topic | Lead |
|---|---|---|
| 1 | XAI for RNN-based traffic matrix prediction (LIME, SHAP, Backprop, IG) | IMT |
| 2 | XAI for Experimental Methodology using RNN | IMT |
| **3** | **XAI for GNN-based KPI prediction — RouteNet Family** | **Dana / Sogeti** |

**This presentation covers Contribution 3.**
The V2X/SDN context remains the thread throughout: in V2X, automated routing and resource allocation decisions must be explainable — a wrong or untransparent decision can have direct safety consequences.

---

## Slide 2 — Phase 1: XAI Survey (DONE — D3.2-Phase1.docx)

**What was delivered:**
A structured survey of XAI methods applicable to AI-driven network automation in SDN/NFV/V2X environments.

**Key sections covered:**
- What is automated in SDN/NFV: routing, traffic steering, VNF placement, resource allocation
- Closed-loop control as the basis for automation (observe → decide → act → measure → adapt)
- V2X as the driving use case: strict latency + reliability requirements, high mobility
- Role of XAI: making RNN/GNN decisions auditable and justifiable in safety-critical contexts
- Survey scope: XAI for RNN and GNN models — perturbation-based (LIME, SHAP), gradient-based (backprop, Integrated Gradients), attention-based

**Output:** Methodological foundation for Phase 2 implementation work.

---

## Slide 3 — Phase 2: Implementation on RouteNet-Fermi (CURRENT)

**Why RouteNet-Fermi?**
RouteNet-Fermi is a state-of-the-art GNN from UPC Barcelona that predicts per-flow network KPIs (delay, jitter, packet loss) from a network simulation. It is the reference GNN for network performance modeling — Contribution 3 applies XAI to this model.

**Goal of Phase 2:**
Apply KernelSHAP to the pretrained delay model → identify which input features drive predictions → mask irrelevant features → validate by retraining.

**8-step work plan:**
| Step | Task | Status |
|------|------|--------|
| 1 | Understand input-output mapping | **DONE** |
| 2 | Run predict.py on all traffic models | **DONE** |
| 3 | Apply KernelSHAP to pretrained model | **DONE** |
| 4 | Analyze importance scores | **DONE** |
| 5 | Mask irrelevant features (zero out) | Next |
| 6 | Compare full vs masked model predictions | Next |
| 7 | Generate modified dataset | Planned |
| 8 | Retrain with reduced input dimensions | Planned |

---

## Slide 4 — Step 1: Complete Input-Output Mapping

**The RouteNet-Fermi input is a heterogeneous graph with 3 entity types and 22 named inputs:**

### Path/Flow features (12 keys — one value per flow)
| Feature | Type | Physical meaning |
|---------|------|-----------------|
| `traffic` | scalar | Average bandwidth of the flow (bps) |
| `packets` | scalar | Packet generation rate (pkts/s) |
| `length` | integer | Number of hops in the route (graph structure) |
| `model` | categorical | Traffic model type (0=CBR, 1=OnOff, … 6=AR2) |
| `eq_lambda` | scalar | Equivalent arrival rate |
| `avg_pkts_lambda` | scalar | Average packets per unit time |
| `exp_max_factor` | scalar | Exponential burst factor (OnOff model param) |
| `pkts_lambda_on` | scalar | Packet rate during ON period |
| `avg_t_off` | scalar | Average OFF duration |
| `avg_t_on` | scalar | Average ON duration |
| `ar_a` | scalar | Autocorrelation coefficient |
| `sigma` | scalar | Std deviation of inter-arrival times |

### Link features (2 keys — one value per network link)
| Feature | Physical meaning |
|---------|----------------|
| `capacity` | Link bandwidth capacity (bps) |
| `policy` | Scheduling policy (WFQ / SP / DRR / FIFO) |

### Queue features (3 keys — one value per queue/buffer)
| Feature | Physical meaning |
|---------|----------------|
| `queue_size` | Buffer size (bytes) |
| `priority` | Priority class (0/1/2) |
| `weight` | WFQ weight |

### Graph topology (5 tensors)
`link_to_path`, `queue_to_path`, `queue_to_link`, `path_to_queue`, `path_to_link`
→ Sparse adjacency tensors encoding which links/queues sit on which flow's route.

**Output:** One delay prediction (seconds) per flow.
→ A 200-node simulation with ~240 flows produces 240 delay values in one forward pass.

**Compact notation:**
```
H = (H_path [F×13], H_queue [Q×5], H_link [L×6], H_routing [sparse]) → y [F×1]
```

---

## Slide 5 — Why We Focused SHAP on 10 Features (not all 22)

**Not all 22 inputs are equally shapeable:**

| Input group | Count | Why excluded from SHAP |
|-------------|-------|------------------------|
| `length` | 1 | Graph structure — changing it means changing the route/topology |
| `model` (categorical) | 1 | One-hot encoded into 7 bits inside the GNN — not a continuous scalar to perturb |
| Link features | 2 | Defined per network link, shared across all flows — not per-flow traffic characteristics |
| Queue features | 3 | Network infrastructure config (buffer sizes, QoS priorities) — not traffic source params |
| Graph tensors | 5 | Connectivity, not features |
| **Path scalars** | **10** | **Continuous, per-flow, directly controllable traffic parameters → SHAP target** |

**The 10 target features are exactly the traffic source parameters**: they describe HOW each flow generates traffic (rate, burst, timing) — the knobs an operator or traffic shaper could change.

**Why constant_bitrate as the baseline?**
It is the simplest traffic model: `exp_max_factor`, `pkts_lambda_on`, `avg_t_off`, `avg_t_on`, `ar_a`, `sigma` are always 0 for CBR (no ON/OFF, no autocorrelation). This gives a clean, unambiguous baseline before moving to complex traffic models.

---

## Slide 6 — Step 2: Model Validation

**What was done:** Ran `predict.py` on all 5 traffic models (200 test samples each), saved predictions vs ground truth.

**MAPE results — matches the paper:**

| Traffic Model | Mean MAPE | Median MAPE | Note |
|---|---|---|---|
| constant_bitrate | 4.29% | 1.15% | Cleanest — good XAI baseline |
| onoff | 2.74% | 0.99% | |
| autocorrelated | **2.46%** | **0.93%** | Best performance |
| modulated | 5.26% | 3.82% | Hardest — high variability |
| all_multiplexed | 4.53% | 2.48% | Mix of all models |

**Conclusion:** The pretrained model is valid. We are explaining a well-performing model, not a broken one.

Note: `ds_test.take(200)` = 200 *network simulations*, not 200 flows. Each simulation contains ~100–500 flows, so the model is actually evaluated on tens of thousands of delay predictions.

---

## Slide 7 — Step 3: Applying KernelSHAP to a GNN

**Challenge:** RouteNet-Fermi takes heterogeneous dict inputs with ragged tensors — standard SHAP libraries cannot perturb them directly (they expect flat feature vectors).

**Our solution — custom KernelSHAP wrapper:**
1. Fix the full graph (all links, queues, other flows, topology) — only the target flow's features change
2. For each SHAP perturbation: replace the 10 scalar path features of flow 0 → call model → record its predicted delay
3. `shap.KernelExplainer` handles the Shapley value computation around this wrapper
4. Background reference = median of each feature across 200 test samples

**Why 40 flows (20 low + 20 high)?**
- KernelSHAP with 256 perturbations per flow takes ~45 seconds per flow on CPU → 40 flows ≈ 30 min total
- We need enough flows for statistically stable mean |SHAP| values — 40 is the practical budget
- Low + high split is a standard contrastive XAI design: it tests whether the model uses *different* features to explain low-delay vs high-delay regimes — a richer finding than a single average

**Why 256 perturbations per flow?**
KernelSHAP accuracy improves with more samples. 256 is a common sweet spot: enough for stable estimates with 10 features (rule of thumb: ≥ 10× the number of features).

---

## Slide 8 — Step 3 Results: Global Feature Importance

**[INSERT: shap_plots/mean_shap_bar.png]**

**Only 2 features dominate for constant_bitrate:**

| Feature | Mean |SHAP| | % of total | Interpretation |
|---------|--------------|------------|---------------|
| traffic | 0.196 s | **49.7%** | Bandwidth load → direct driver of delay |
| packets | 0.186 s | **47.0%** | Packet rate → determines packet size (traffic/packets) |
| avg_pkts_lambda | 0.008 s | 1.95% | Small but non-zero |
| eq_lambda | 0.005 s | 1.34% | Small but non-zero |
| 6 timing params | 0.000 s | **0%** | **Completely irrelevant for CBR** |

**Physical explanation for the bottom 6 being zero:**
CBR traffic has no ON/OFF periods, no autocorrelation, no burst factor — these 6 features are *always exactly 0* in the constant_bitrate dataset. SHAP correctly identifies them as non-explanatory.

---

## Slide 9 — Step 3 Results: Low vs High Delay Regimes

**[INSERT: shap_plots/low_vs_high_shap.png]**

Delay range in test set: 0.206s (low) to 0.434s (high)

**Key finding — importance shifts at high delay:**

| Feature | Low delay | High delay | Ratio |
|---------|-----------|-----------|-------|
| traffic | 0.224 s | 0.168 s | 0.75x (less critical) |
| packets | 0.220 s | 0.152 s | 0.69x |
| **eq_lambda** | 0.001 s | **0.010 s** | **9x more important** |
| **avg_pkts_lambda** | 0.003 s | **0.013 s** | **5x more important** |

**Interpretation:**
- At *low delay*: the system is lightly loaded — raw bandwidth (`traffic`) is the dominant factor.
- At *high delay*: the network approaches saturation — arrival rate parameters (`eq_lambda`, `avg_pkts_lambda`) begin influencing queuing behavior. This is consistent with queuing theory (M/M/1 queue: as ρ→1, λ becomes critical).

---

## Slide 10 — Step 3 Results: Beeswarm & Single-Flow Waterfall

**[INSERT: shap_plots/shap_summary_dot.png]**

Each dot = one flow. Color = feature value (blue=low, pink=high).

- `traffic` (pink dots): high bandwidth → positive SHAP → more delay ✓
- `packets` (pink dots): high packet rate → **negative** SHAP → LESS delay
  → Because `packet_size = traffic / packets`: more packets at same bandwidth = smaller packets = less *transmission delay*
- Bottom 6: all dots collapsed at zero — confirmed irrelevant

**[INSERT: shap_plots/shap_waterfall_worst.png]**

Worst-case flow (measured delay = 0.434s, model base = 0.407s):
- `traffic` pushes **+0.104s** (very high bandwidth → heavy congestion)
- `packets` pulls **−0.082s** (high packet rate → small packets → lower transmission delay)
- Net contribution: +0.027s above base → physically coherent

---

## Slide 11 — Step 4: Importance Analysis & Masking Decision

**[INSERT: shap_plots/cumulative_importance.png]**

The cumulative importance curve crosses 99% after 4 features.

**Masking recommendation for constant_bitrate:**

| Feature | Importance | Decision | Reason |
|---------|-----------|----------|--------|
| traffic | 49.7% | **KEEP** | Primary delay driver |
| packets | 47.0% | **KEEP** | Controls packet size (transmission delay) |
| avg_pkts_lambda | 1.95% | **KEEP** | Gains importance at high delay (5x ratio) |
| eq_lambda | 1.34% | **KEEP** | Gains importance at high delay (9x ratio) |
| exp_max_factor | 0% | **MASK** | Zero in CBR — no burst factor |
| pkts_lambda_on | 0% | **MASK** | Zero in CBR — no ON periods |
| avg_t_off | 0% | **MASK** | Zero in CBR — no OFF periods |
| avg_t_on | 0% | **MASK** | Zero in CBR — no ON periods |
| ar_a | 0% | **MASK** | Zero in CBR — no autocorrelation |
| sigma | 0% | **MASK** | Zero in CBR — no variance |

**Result: 10 → 4 path features (60% input reduction)**

Note: `eq_lambda` and `avg_pkts_lambda` are kept despite small global importance because they are *disproportionately important at high delay* (the critical operating regime for V2X safety).

---

## Slide 12 — Next Steps (Steps 5–8)

**Step 5 — Feature Masking (immediate next)**
Zero out the 6 irrelevant features in the existing dataset without retraining. Run the pretrained model on masked inputs and measure the MAPE change.

*Expected result:* MAPE should not degrade (or barely) for constant_bitrate, since the masked features are always 0 anyway.

**Step 6 — Full vs Masked Comparison**
Side-by-side analysis of predictions: plot full vs masked predicted delay, compute MAPE difference per flow. Check whether high-delay flows are impacted more than low-delay flows.

**Step 7 — Generate Modified Dataset**
Create a new version of the dataset with only 4 path features (traffic, packets, eq_lambda, avg_pkts_lambda). This requires modifying `data_generator.py` to drop the 6 zero features from the input dict.

**Step 8 — Retrain with Reduced Input Dimensions**
Modify the `path_embedding` layer input size (currently 10 + 7 = 17 → new: 4 + 7 = 11). Retrain from scratch and compare MAPE vs the original model.

*Scientific question answered by Step 8:* Does reducing the path feature space from 10 to 4 preserve model performance? If yes, the XAI analysis has successfully identified a more efficient model architecture.

---

## Slide 13 — Positioning within TRAVEL / WP3

```
TRAVEL WP3 — XAI for Network Automation
│
├── Task 3.2 (SogetiLabs + IMT-NE)
│   │
│   ├── Phase 1 ✓  Survey of XAI methods for RNN/GNN (D3.2-Phase1.docx)
│   │             [Perturbation: LIME/SHAP | Gradient: Backprop/IG | Attention]
│   │
│   └── Phase 2 (current)
│       ├── Contribution 1+2: XAI for RNN traffic prediction (IMT)
│       └── Contribution 3: XAI for GNN-based KPI prediction (Sogeti/Dana)
│           ├── Steps 1-4 ✓  KernelSHAP on RouteNet-Fermi delay model
│           └── Steps 5-8    Masking → validation → retraining
│
└── V2X context: every result is interpreted through the lens of
    latency-sensitive, safety-critical vehicular network automation
```

**Contribution to D3.2:**
This work produces the first application of KernelSHAP to a GNN-based network KPI predictor, with a concrete methodology for identifying and pruning irrelevant traffic features in SDN/V2X scenarios.

---

## Slide 14 — Summary & Key Results

**What has been accomplished:**

1. **Full input-output mapping** of RouteNet-Fermi: 22 named inputs across path/link/queue entities + 5 graph topology tensors → per-flow delay

2. **Model validated** on 5 traffic models — MAPE 2.5%–5.3%, matching the paper

3. **KernelSHAP applied** to a GNN for the first time in this project:
   - Custom wrapper to perturb per-flow scalar features while keeping the graph intact
   - 40 flows × 256 perturbations each
   - 4 plots generated (bar chart, low vs high, beeswarm, waterfall)

4. **Feature importance quantified:**
   - `traffic` (50%) + `packets` (47%) + `avg_pkts_lambda` (2%) + `eq_lambda` (1%) = 100%
   - 6 timing parameters = zero importance for CBR → candidates for masking

5. **Masking plan defined:** 10 → 4 path features (60% reduction), justified by SHAP + queuing theory

**Next:** Steps 5–8: masking validation → retraining → scientific conclusion
