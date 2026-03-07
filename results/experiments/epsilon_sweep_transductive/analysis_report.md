# Epsilon Impact Analysis Report
## DP-MLP Pretraining on ogbn-arxiv
**Date:** 2026-03-07T09:13:14
**Non-DP Baseline Test Accuracy:** 47.57%

## Results Summary

| Epsilon | Test Acc | Val Acc | Train Acc | Steps | Gap to Non-DP | % of Non-DP |
|---------|----------|---------|-----------|-------|---------------|-------------|
| 0.13 | 0.74% | 0.88% | 1.64% | 1 | 46.83% | 1.6% |
| 0.51 | 0.87% | 1.01% | 1.60% | 24 | 46.70% | 1.8% |
| 1.00 | 0.96% | 1.22% | 2.02% | 92 | 46.61% | 2.0% |
| 2.02 | 1.10% | 1.23% | 1.71% | 37 | 46.47% | 2.3% |
| 5.01 | 5.87% | 7.63% | 17.86% | 241 | 41.70% | 12.3% |
| 10.03 | 5.88% | 7.66% | 17.69% | 107 | 41.69% | 12.4% |
| 50.09 | 5.89% | 7.66% | 17.90% | 124 | 41.68% | 12.4% |
| 101.22 | 5.87% | 7.64% | 17.91% | 43 | 41.70% | 12.3% |
| Non-DP | 47.57% | - | - | - | 0.00% | 100.0% |

## Key Findings

1. **Best DP Performance:** ε=50.09 achieves 5.89% test accuracy
2. **Epsilon Impact:** Increasing ε from 0.13 to 101.22 improves accuracy by 5.13%
3. **Trend Validation:** ✓ Higher epsilon leads to better performance (as expected)
4. **Gap to Non-DP:** Even at highest epsilon (101.22), DP performance is 41.70% below non-DP baseline

## Hyperparameters Used

- **Noise Multiplier:** 5.0
- **Batch Size:** 10000
- **Learning Rate:** 0.003
- **Epochs:** 20
- **Latent Size:** 100
- **Num Layers:** 3

## Recommendations

1. **Hyperparameter Tuning:** Consider increasing batch size and reducing noise multiplier for better utility-privacy trade-off
2. **Training Duration:** DP training may require more epochs to converge
3. **Alternative Approaches:** Consider DP-Adam or other DP optimizers
4. **Lower Epsilon Ranges:** The tested epsilon values may be too high; consider exploring ε < 1
