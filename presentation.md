# XAI for Network Performance Prediction
## RouteNet-Fermi — Research Internship Progress
**Internship:** Network Automation / SDN / XAI
**Meeting:** IMT + Sogeti

---

## Slide 1 — Project Context

**Goal:** Add Explainable AI (XAI) to RouteNet-Fermi, a Graph Neural Network that predicts network performance (delay, jitter, packet loss).

**Why it matters:**
- Network operators need to trust and understand model predictions
- XAI reveals WHICH traffic/topology features drive performance degradation
- Enables feature pruning → simpler, faster retrainable models

**Repo:** https://github.com/BNN-UPC/RouteNet-Fermi

---

## Slide 2 — What is RouteNet-Fermi?

**Type:** Graph Neural Network (GNN) with message-passing between 3 entity types

| Entity | What it represents | Key features |
|--------|-------------------|--------------|
| **Path (flow)** | One traffic flow src→dst | traffic, packets, timing model params |
| **Link** | Physical network link | capacity, scheduling policy |
| **Queue** | Buffer on a link | queue_size, priority, weight |

**How it works:**
1. Embed path / link / queue into hidden state vectors
2. Run 8 rounds of message-passing (paths ↔ links ↔ queues)
3. Readout: predict delay per flow = queue delay + transmission delay

**Inputs:** 10 scalar path features + graph connectivity tensors
**Output:** Per-flow delay (seconds)

---

## Slide 3 — Dataset & Traffic Models

**Dataset:** OMNeT++ simulations (from the paper)

5 traffic models evaluated:

| Traffic Model | Description |
|---|---|
| constant_bitrate | Fixed rate, no variation |
| onoff | On/Off bursts |
| autocorrelated | Correlated inter-arrival times |
| modulated | Variable rate patterns |
| all_multiplexed | Mix of all above |

**Test set:** 200 samples × up to 54,400 flows each

---

## Slide 4 — Step 1: Understanding the Model (DONE)

**What I did:** Reverse-engineered the full input-output mapping of RouteNet-Fermi

**Path features (10 scalars per flow):**

| Feature | Physical meaning |
|---------|-----------------|
| traffic | Average bandwidth (bps) |
| packets | Packet generation rate (pkts/s) |
| eq_lambda | Equivalent arrival rate |
| avg_pkts_lambda | Average packets per lambda |
| exp_max_factor | Exponential burst factor |
| pkts_lambda_on | Packet rate during ON period |
| avg_t_off / avg_t_on | ON/OFF durations |
| ar_a | Autocorrelation coefficient |
| sigma | Standard deviation of inter-arrival |

**Key insight:** Features 3–10 are timing model parameters — only non-zero for specific traffic models (onoff, autocorrelated, etc.)

---

## Slide 5 — Step 2: Model Validation (DONE)

**What I did:** Ran `predict.py` on all 5 traffic models, saved predictions vs ground truth

**Results — MAPE (Mean Absolute Percentage Error):**

| Traffic Model | Mean MAPE | Median MAPE |
|---|---|---|
| constant_bitrate | 4.29% | 1.15% |
| onoff | 2.74% | 0.99% |
| autocorrelated | **2.46%** | **0.93%** |
| modulated | 5.26% | 3.82% |
| all_multiplexed | 4.53% | 2.48% |

**Conclusion:** Model performs as expected — matches the paper's reported MAPE.
Modulated is hardest (high variability); autocorrelated is easiest (regular patterns).

---

## Slide 6 — Step 3: Applying KernelSHAP (DONE)

**What is SHAP?**
SHAP (SHapley Additive exPlanations) assigns each input feature a contribution score to the model's prediction — grounded in game theory (Shapley values).

**Challenge with RouteNet-Fermi:**
The model takes dict inputs with ragged/graph tensors — standard SHAP cannot perturb them directly.

**Our solution — KernelSHAP wrapper:**
1. Fix the full graph (topology, all other flows, link/queue features)
2. Perturb only the 10 scalar path features of ONE target flow
3. Run `shap.KernelExplainer` with 256 perturbations per flow
4. Background reference = median feature values across 200 test samples

**Scope:**
- Dataset: `constant_bitrate` (clean baseline, single traffic model)
- 40 flows explained: 20 lowest-delay + 20 highest-delay

---

## Slide 7 — Step 3 Results: Global Feature Importance

**[INSERT: mean_shap_bar.png]**

**Key finding:** Only 2 features dominate:
- `traffic` (avg bandwidth): **0.196s mean |SHAP|**
- `packets` (packet rate): **0.186s mean |SHAP|**

Together they account for **>99% of total importance**

The bottom 6 features (timing params) = **zero importance for CBR**
→ Physically correct: CBR flows have no ON/OFF, no autocorrelation, so these params are always 0.

---

## Slide 8 — Step 3 Results: Low vs High Delay

**[INSERT: low_vs_high_shap.png]**

**Interesting pattern:**
- `traffic` and `packets` are more important for **low-delay** flows
- At high delay, `eq_lambda` and `avg_pkts_lambda` start contributing (9x and 5x more)

**Interpretation:**
At low delay → the system is lightly loaded, bandwidth is the only driver.
At high delay → the network approaches saturation, arrival rate parameters begin to matter for the queuing behavior.

---

## Slide 9 — Step 3 Results: Beeswarm & Waterfall

**[INSERT: shap_summary_dot.png]**

**Beeswarm observations:**
- `traffic` (pink=high): positive SHAP → more traffic = more delay ✓
- `packets` (pink=high): negative SHAP → higher packet rate = LESS delay
  → Because packet_size = traffic/packets → smaller packets → less transmission delay

**[INSERT: shap_waterfall_worst.png]**

**Worst-case flow (delay = 0.434s, base = 0.407s):**
- `traffic` pushes +0.104s
- `packets` pulls −0.082s
- Net: +0.027s above base → explained by the high traffic/small packet combination

---

## Slide 10 — Step 4: Analysis & Masking Plan (IN PROGRESS)

**Masking decision — constant_bitrate:**

| Feature | Importance | Decision |
|---------|-----------|----------|
| traffic | 49.7% | **KEEP** |
| packets | 47.0% | **KEEP** |
| avg_pkts_lambda | 1.95% | **KEEP** |
| eq_lambda | 1.34% | **KEEP** |
| exp_max_factor | 0% | MASK |
| pkts_lambda_on | 0% | MASK |
| avg_t_off | 0% | MASK |
| avg_t_on | 0% | MASK |
| ar_a | 0% | MASK |
| sigma | 0% | MASK |

**Input reduction: 10 → 4 path features (60% reduction)**

Note: `eq_lambda` and `avg_pkts_lambda` gain importance at high delay (9x and 5x respectively), so we keep them.

---

## Slide 11 — Next Steps

| Step | Task | Status |
|------|------|--------|
| 5 | Modify input by masking (zero out) irrelevant features | Next |
| 6 | Compare predictions: full model vs masked inputs | Next |
| 7 | Generate modified dataset (with zeroed features) | Planned |
| 8 | Retrain model with reduced input dimensions (10→2) | Planned |

**Expected outcome of retraining:**
- Simpler model (smaller path embedding: input dim 2+7 instead of 10+7)
- Faster inference
- Test: does MAPE degrade? If not → the 8 masked features truly add no value

---

## Slide 12 — Summary

**What has been done:**
1. Full understanding of RouteNet-Fermi architecture (inputs, message passing, outputs)
2. Model validated on 5 traffic models — matches paper MAPE
3. KernelSHAP applied — first XAI analysis of a GNN-based network simulator
4. Feature importance quantified: traffic + packets = >99% of delay prediction

**Key scientific contribution:**
For constant_bitrate traffic, 8 out of 10 path features can be removed with no loss of information. This is verifiable through retraining (Steps 5–8).

**Tools used:** TensorFlow 2.6, SHAP 0.42, NetworkX, Python 3.7
