# Engression Simulation Study

This repository implements a complete simulation framework to compare **Engression loss functions** under various data-generating mechanisms. The study evaluates five losses—`power_0.5`, `power_1`, `exp`, `log1p`, and `frac`—across **pre-ANM** and **post-ANM** models to assess accuracy, robustness, and stability of conditional mean estimation.

---

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/seyonghw/Engression-Simulation.git  
cd Engression-Simulation

### 2. Create and activate a virtual environment
python3 -m venv .venv  
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

### 3. Install dependencies
pip install -r requirements.txt

Dependencies:
torch  
pandas  
numpy  
matplotlib  
seaborn  
tqdm  
pytest

---

## Run the Complete Analysis

All steps are automated with the provided **Makefile**.

make all

This runs the following pipeline:

| Step | Script | Description |
|------|---------|-------------|
| 1 | test.py | Runs the full 72×10 simulation and saves result/mse_engression_losses.csv |
| 2 | plot_results.py | Generates boxplots of mean MSEs per (model, dx,dy,k) |
| 3 | plot_loss_distributions.py | Produces violin, density, and ECDF plots for MSE distributions |
| 4 | outlier.py | Summarizes high-MSE cases (mse > 1) by frequency and magnitude |

Outputs are written to:
result/  
└── figures/

You can also run individual targets:
make simulate        # run only the simulation  
make figures         # build all plots  
make outliers        # summarize high-MSE cases

---

## Estimated Runtime

| Environment | Runtime |
|--------------|----------|
| CPU (standard laptop) | ~14 hr |
| GPU (CUDA) |  |

---


# Analysis & Key Findings

### 1. Distribution of MSE
Violin and density plots show:
- frac consistently yields the lowest and tightest MSE distribution.
- power_0.5, exp, and log1p have similar medians but broader spreads.
- Outlier tails are thinner for frac, indicating greater stability.

### 2. Boxplots by Model & Dimension
| Model | (dx,dy,k) | Observation |
|:--|:--|:--|
| pre-ANM (1,1,1) | All losses similar; frac slightly better |
| pre-ANM (2,2,2) | Hardest case; exp and log1p show occasional large MSEs |
| post-ANM (any) | Overall easier; all losses perform comparably |

### 3. Outlier Summary
- Most MSE > 1 cases occur under pre-ANM + cubic + high noise (σ = 2).  
- exp and log1p losses show higher frequency of outliers.  
- frac rarely exceeds MSE = 1.

### 4. Rankings (Mean rank over 72 cases)
| Loss | Mean Rank | Performance |
|:--|:--|:--|
| frac | 1.3 | Best overall |
| power (β=0.5) | 2.2 | Strong alternative |
| exp | 2.8 | Less robust under high noise |
| log1p | 3.0 | Slightly unstable in input-noise cases |

### 5. Main Insights
- Fractional loss (frac) provides the most stable and accurate estimation overall.  
- Input-noise (pre-ANM) models are harder and highlight robustness differences.  
- Bounded losses (like frac) handle heteroskedasticity better than unbounded ones (exp, log1p).

---

# Automated Testing

Located in tests/test_section7.py:
1. Support & shape checks — ensures all generators respect input domains.  
2. Closed-form verification — for post-ANM with zero noise: Y = M h(Ax).  
3. Variance structure test — confirms pre-ANM is heteroskedastic, post-ANM is homoskedastic.

Run:
pytest -q

---

# Makefile Automation

Core targets:

| Command | Description |
|----------|--------------|
| make all | Run simulation → plots → outlier analysis |
| make simulate | Generate mse_engression_losses.csv |
| make figures | Build all figures |
| make boxplots | Boxplots of mean MSE per (model, dims) |
| make distributions | Violin/density/ECDF plots |
| make outliers | Create high-MSE summary tables |
| make clean | Remove generated figures and summaries |

---

# Summary of Findings

- frac loss consistently achieves the lowest and most stable MSE across all scenarios.  
- Unbounded losses (exp, log1p) are sensitive to strong nonlinearity and large noise.  
- Pre-ANM models reveal robustness differences; post-ANM models are easier and show small performance gaps.  
- Overall, bounded energy-based losses improve stability in heteroskedastic, nonlinear data settings.

---

