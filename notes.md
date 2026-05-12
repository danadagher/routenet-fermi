# Worked on traffic_models

# Used take(200) on test set because full set was too slow for SHAP iteration 

## After the execution of predict.py for delay prediction intraffic_models

Ran predict.py on 200 test samples × 5 traffic models = 54,400 flows each.
Saved predictions, labels, flow_counts as .npy files.

MAPE results:
| Traffic model      | Mean  | Median |
|--------------------|-------|--------|
| constant_bitrate   | 4.29% | 1.15%  |
| onoff              | 2.74% | 0.99%  |
| autocorrelated     | 2.46% | 0.93%  |
| modulated          | 5.26% | 3.82%  |
| all_multiplexed    | 4.53% | 2.48%  |

Model is performing as expected (matches paper's reported MAPE).
modulated is hardest, autocorrelated is easiest.


# Applying SHAP to delay(one choosen sub-dataset) traffic model only; extension to jitter/loss is mechanical
 
# Use all_multiplexed sub-dataset when Appling Graph SHAP on the pretrained model and Analyzing the imporatance scores of the inputs
# why ? it's the only one where model varies, so it's the only one where SHAP can attribute importance to traffic model identity


## Step 3 — plan for tomorrow

Apply SHAP to delay model only.
Wrapper strategy: pick one flow per sample, perturb its 12 path features,
use shap.KernelExplainer with median-feature background.

Targets:
- 20 low-delay flows (delay < 0.2s)
- 20 high-delay flows (delay > 1.0s)
- Compare feature importance between regimes

# @RoutNet-Fermi > @traffic_models > delay > all_multiplexed .
