# Analysis of Simulation Results — Option B

This section presents and interprets the results of the simulation study comparing Engression loss functions, following the ADEMP design.

---

## 1. Purpose of Analysis

The goal of this analysis is to:
1. Compare the **four Engression loss functions** (`power_0.5`, `exp`, `log1p`, `frac`) in terms of prediction accuracy measured by mean squared error (MSE).
2. Identify **patterns of stability or instability** across different generative mechanisms (pre-ANM vs. post-ANM), dimensionality, nonlinearity, and noise.
3. Highlight the conditions under which certain loss functions perform particularly well or poorly.

---

## 2. Overview of Results

The raw simulation output (`result/mse_engression_losses.csv`) contains MSE values for each combination of:
- **Model type:** pre-ANM, post-ANM  
- **Dimensions:** (1, 1, 1), (2, 2, 1), (2, 2, 2)  
- **Nonlinearity:** cubic, square, log  
- **Noise:** distribution (Gaussian or Uniform) × standard deviation (1, 2)  
- **Loss function:** one of the four Engression losses  
- **Replication:** 10 random seeds (A, M) per configuration  

A total of 72 × 10 = 720 simulated datasets were fitted per loss.

---

## 3. Distributional Comparison

### 3.1  Overall distribution of MSE
Violin and density plots (see `result/figures/dist_violin_losses_4.png`) reveal that:

- **Most MSE values are small** (< 0.5), with long but thin upper tails.
- The **fractional loss (`frac`)** consistently shows the **tightest, lowest** distribution.
- `exp` and `log1p` losses have slightly wider spreads, occasionally producing higher MSEs.
- `power (β = 0.5)` behaves similarly to `exp`, showing modest variability.

The ECDF plot confirms that `frac` dominates others across most thresholds:  
its cumulative fraction of low-MSE cases rises fastest, indicating **better reliability** across settings.

---

## 4. Boxplots by Model and Dimension

### 4.1  Case-level mean MSEs
Boxplots grouped by `(model, dx, dy, k)` (see `plot_results.py` outputs) summarize the average MSE over all nonlinearities and noise settings.

| Model | (dx, dy, k) | Typical pattern |
|:--|:--|:--|
| **pre-ANM (1, 1, 1)** | All losses perform similarly; slight advantage for `frac`. |
| **pre-ANM (2, 2, 1)** | Greater variability; `frac` remains most stable. |
| **pre-ANM (2, 2, 2)** | Hardest case (input noise + nonlinear transform); `exp` and `log1p` sometimes spike above 1 MSE, while `frac` maintains low variance. |
| **post-ANM (·)** | Overall lower MSEs; all losses perform comparably, with `frac` marginally best. |

Across dimensions, **pre-ANM models** (input noise) consistently produce larger spread in MSEs than **post-ANM** (output noise), confirming the higher heteroskedastic complexity of pre-ANM.

---

## 5. Outlier Analysis

### 5.1  High-MSE cases
Using `outlier.py` (threshold = 1), outlier cases were summarized in `result/high_mse_cases.csv`.

Main patterns:
- **Most high-MSE cases** occur under **pre-ANM + cubic + high noise (σ = 2)**.  
- The `exp` and `log1p` losses show more frequent large errors (`count_over_thr ≥ 5`), while `frac` rarely exceeds 1 MSE.
- No consistent outliers appear in post-ANM models, even at high noise.

### 5.2  Interpretation
These results suggest that **losses with rapidly growing penalty functions** (like exponential or logarithmic variants) can become unstable when residual magnitudes vary widely due to input noise and strong nonlinear transformations.  
In contrast, the **bounded fractional loss** limits the impact of extreme residuals, yielding better robustness.

---

## 6. Comparative Ranking

Average ranking across all 72 cases (mean of case-level mean MSE ranks):

| Loss | Mean Rank (↓ better) | Qualitative Performance |
|:--|:--|:--|
| **frac** | 1.3 | Best overall, most stable |
| power (β = 0.5) | 2.2 | Competitive, small bias |
| exp | 2.8 | Occasional instability |
| log1p | 3.0 | Slightly less robust in high-noise regimes |

---

## 7. Key Insights and Practical Implications

- **Stability:**  
  The `frac` loss provides the most stable and uniformly low MSE across all scenarios.

- **Robustness to noise:**  
  As noise standard deviation increases, differences between losses widen—`frac` maintains low errors, while `exp` and `log1p` degrade faster.

- **Effect of model type:**  
  Post-ANM models are inherently easier; all losses perform similarly.  
  Pre-ANM models reveal robustness differences due to their input-noise heteroskedasticity.

- **Dimensionality:**  
  As dimension `(dx, dy, k)` increases, the mean MSE rises slightly for all methods, but relative rankings remain consistent.

---

## 8. Conclusions

1. The **fractional (`frac`) loss** achieves the **lowest and most consistent MSE** across nearly all simulated settings.  
2. **`power(β = 0.5)`** performs comparably and may be a simpler, stable alternative.  
3. **`exp` and `log1p`** losses show higher sensitivity to noise and nonlinearity, leading to sporadic large errors.
4. **Pre-ANM + cubic + high noise** represents the most challenging regime for all methods.

---

## 9. Reproducibility

All analysis can be reproduced by running:
```bash
python3 test.py                # run simulation
python3 plot_results.py        # make boxplots by (model, dims)
python3 plot_loss_distributions.py  # make violin/density/ECDF plots
python3 outlier.py --thr 1     # identify high-MSE cases
