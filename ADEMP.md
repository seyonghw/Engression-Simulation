# ADEMP Framework for Engression Simulation Study

This document summarizes the simulation design using the **ADEMP framework** (Aims, Data-generating mechanism, Estimands, Methods, Performance measures).

---

## A — Aims

The goal of this simulation study is to evaluate and compare the **performance of five loss functions** used in the **Engression** framework under varying data-generating mechanisms.

### Key questions
- How do different losses perform under **pre-ANM** and **post-ANM** models (input-noise vs. output-noise structures)?
- How sensitive are the losses to the **nonlinearity strength** of the true function and **noise level**?
- Which loss achieves the **lowest and most stable conditional mean estimation error** across diverse scenarios?

---

## D — Data-generating Mechanism

### Model forms
Two data-generating processes are considered:

- **Pre-ANM:**  
  \( Y = M\,h(A(X + \varepsilon_x)) \)
- **Post-ANM:**  
  \( Y = M\,h(A X) + \varepsilon_y \)

Here, \( X \in \mathbb{R}^{d_x} \), \( Y \in \mathbb{R}^{d_y} \),  
and \( A, M \) are random linear mixing matrices drawn once per replication.

### Nonlinearity \( h(\cdot) \)
Each element of \( A X \) passes through one of three nonlinearities:
1. **Cubic:** \( h(z) = z^3 / 3 \)
2. **Square (ReLU²/2):** \( h(z) = \mathrm{ReLU}(z)^2 / 2 \)
3. **Piecewise log:**  
   \( h(z) = \log(1 + z) \) for \( z>2 \),  
   otherwise linear-log transition for smoothness.

### Parameters varied
| Factor | Values | Description |
|:--|:--|:--|
| **Model type** | pre-ANM, post-ANM | Input vs output noise |
| **Dimensions (d_x, d_y, k)** | (1,1,1), (2,2,1), (2,2,2) | Latent-space and observation sizes |
| **Nonlinearity** | cubic, square, log | Different curvature strengths |
| **Noise distribution** | Gaussian, Uniform | Shape of random noise |
| **Noise standard deviation** | 1, 2 | Noise magnitude |

- **Total design points:** \( 2 \times 3 \times 3 \times 2 \times 2 = 72 \)
- **Replications:** 10 per setting (new random \( A, M \))
- **Training sample size:** \( n = 10{,}000 \)
- **Evaluation grid:** 50×50 (2D) or 100 points (1D)

---

## E — Estimands

The target quantity is the **conditional mean function**
\[
\mu(x) = \mathbb{E}[Y \mid X = x],
\]
which is approximated via Monte-Carlo averaging from the true data generator.  
Each Engression model aims to estimate this function under its chosen loss.

---

## M — Methods

Each simulated dataset is fitted using **Engression** models trained with the following loss functions:

| Loss | Label in code | Description |
|:--|:--|:--|
| Power (β = 0.5) | `power_0.5` | Energy distance with fractional power |
| Power (β = 1) | `power_1` | Standard energy (L¹) distance |
| Exponential | `exp` | Smooth exponential energy variant |
| Logarithmic | `log1p` | Log(1+distance) energy variant |
| Fractional | `frac` | Bounded fractional energy distance |

Training setup:
- Learning rate = 0.005  
- Batch size = 1000  
- Epochs = 300  
- Device = CPU/GPU (PyTorch)

The main comparison focuses on the **four losses** (`power_0.5`, `exp`, `log1p`, `frac`), excluding `power_1` in most analyses.

---

## P — Performance Measures

### Primary metric
**Mean Squared Error (MSE):**
\[
\text{MSE} = \frac{1}{|X_{\text{grid}}|} \sum_x \| \hat{Y}(x) - Y_{\text{true}}(x) \|^2,
\]
computed on the evaluation grid.

### Summaries and plots
1. **Case-level mean MSE:** averaged over 10 replications for each configuration.  
2. **Box plots:** distributions of case-level mean MSE per loss for each `(model, dx,dy,k)` combination.  
3. **Violin / ECDF plots:** overall MSE distributions across all cases to compare the full shape of errors.  
4. **Outlier analysis:** cases with MSE > 1 summarized by frequency (`count_over_thr`) and magnitude (`mse_mean_over_thr`).

### Interpretation goals
- Identify the loss function achieving the lowest and most stable MSE across models and nonlinearities.  
- Diagnose instability regimes (e.g., pre-ANM + cubic + high noise) where certain losses degrade.

---

## ADEMP Summary Table

| Element | Description |
|:--|:--|
| **Aims** | Compare Engression loss functions on accuracy and stability under diverse generative models |
| **Data-generating** | 72 cases varying dimensions, nonlinearity, noise type & level; 10 replications each |
| **Estimands** | Conditional mean \( \mathbb{E}[Y|X] \) |
| **Methods** | Engression with five loss functions; main comparison uses four (`power_0.5`, `exp`, `log1p`, `frac`) |
| **Performance** | MSE on test grid; summarized via boxplots, violin plots, and outlier frequency tables |

---

**Summary:**  
This simulation quantifies how the choice of loss function affects Engression’s accuracy and robustness across complex nonlinear data models.  The results show that while all losses perform similarly in low-noise, low-dimensional settings, the **fractional loss (`frac`)** tends to yield the most stable and lowest MSE across challenging high-noise or highly nonlinear regimes.
