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

## Optimized Parallel Execution

### Overview
The optimized version (test_optimized.py) introduces **CPU multiprocessing**:
- Each replication is **independent**
- Parallelizes across REPS
- Achieves **1.4~1,7 speedup** for typical workloads
- Falls back to **serial mode** on GPU to avoid device contention
- Guarantees deterministic results using per-replication seeds

### Run potimized simulation
make simulate_opt

Output:
result/mse_enegression_losses_optimized.csv

---

## Profiling and Optimization Analysis

### Baseline profiling
Runs complexity_profile.py:
make profile

Output:
result/complexity_grid.csv

### Parallel profiling

Runs complexity_profile_optimized.py:
make profile_opt

Output:
result/complexity_grid_optimized.csv

### Bottleneck profiling
Identify time spent in:
- matrix generation
- data simulation
- true-surface computation
- Engression training
- prediction

Run:
make bottleneck

Output:
result/bottleneck_times.csv

---

## Performance Comparison Plots
We provide a script to visualize:
- baseline vs optimized wall-time
- per-grid speedup values
- regression consistency between MSE results

Run:
make perfplots

Generates:
result/figures/runtime_scatter.png
result/figures/speedup_bar.png

Interpretation:
- Points below diagonal -> optimized version is faster
- Bars > 1 indicate speedup (1.5~1.7 for large grids)

---

## Regression Tests (Correctness Check)

Since the optimized pipeline changes **only execution order**, results must remain identical.

Use:
make regression_test

This checks:
- both contain identical (model, dims, noise, loss, seed) rows
- MSE differences are < 1e-6 absolute and relative
- exits with error if any mismatch is detected

Ensures **functional correctness** of optimization.


# Analysis & Key Findings

### **1. Distribution of MSE**
- `frac` achieves the **lowest median MSE** and the **tightest distribution**.
- `power_0.5`, `exp`, and `log1p` have similar centers but noticeably wider variance.
- Outlier tails (large MSE values) are **significantly thinner** for `frac`.

### **2. Model Behavior**
- **pre-ANM (2,2,2)** is the most difficult scenario; nonlinear coupling amplifies noise.
- In these hard regimes, `exp` and `log1p` occasionally produce very large MSE spikes.
- **post-ANM** cases are easier overall, showing small differences between losses.

### **3. Outlier Summary**
- High-MSE cases (MSE > 1) mostly occur under:
  - pre-ANM  
  - cubic nonlinearity  
  - high noise (`σ = 2` or `3`)  
- `exp` and `log1p` generate the largest number of outliers.
- `frac` almost never produces MSE > 1, showing strong robustness.

### **4. Loss Ranking (Mean rank over 72 cases)**
| Loss | Mean Rank | Interpretation |
|------|-----------|----------------|
| `frac` | **1.3** | Best overall; most robust and accurate |
| `power_0.5` | 2.2 | Strong performance, second-best |
| `exp` | 2.8 | Sensitive to noise, unstable in difficult cases |
| `log1p` | 3.0 | Least stable, heavy-tailed errors in pre-ANM |

### **5. Key Insights**
- **Bounded** losses (especially `frac`) handle heavy-tailed, nonlinear noise much better than unbounded ones (`exp`, `log1p`).  
- pre-ANM models reveal robustness differences—**an essential stress-test** for loss functions.  
- `frac` combines **accuracy + stability**, making it the most reliable across all settings.



---

# Automated Testing (pytest)

All correctness and consistency checks are located in `tests/`.

The test suite includes:

### **1. Support & Shape Validation**
Ensures all data generators (`preANM`, `postANM`) produce:
- correct tensor shapes,
- valid ranges for inputs,
- consistent batch dimensions.

### **2. Closed-Form Verification**
For **post-ANM** with zero noise:
\[
Y = M\,h(Ax)
\]
The tests confirm the implementation matches this exact analytical form.

### **3. Variance Structure Tests**
- **pre-ANM** must be heteroskedastic (input noise propagates through `h`).  
- **post-ANM** must be homoskedastic (noise added after transformation).  
The tests check empirical variance patterns to ensure correctness.

### **Running the full test suite**
```bash
pytest -q
```

# Makefile Commands

The project includes a full automation pipeline via `Makefile`.  
Below is the complete list of supported commands:

| Command | Description |
|--------|-------------|
| **make all** | Run the full baseline pipeline (simulation → figures → summaries) |
| **make simulate** | Run baseline simulation (`test.py`) → saves `mse_engression_losses.csv` |
| **make simulate_opt** | Run optimized parallel simulation (`test_optimized.py`) |
| **make figures** | Build all figures (boxplots + distributions) |
| **make boxplots** | Generate boxplots of mean MSE per (model, dims) |
| **make distributions** | Generate violin/density/ECDF distribution plots |
| **make outliers** | Create summaries of high-MSE cases (mse > 1) |
| **make profile** | Profile baseline runtime (`complexity_profile.py`) |
| **make profile_opt** | Profile optimized runtime (`complexity_profile_optimized.py`) |
| **make bottleneck** | Measure bottlenecks (`bottleneck_profile.py`) |
| **make perfplots** | Generate performance comparison plots (baseline vs optimized) |
| **make regression_test** | Validate that optimized results match baseline results |
| **make clean** | Remove generated figures & summaries (keep main CSVs) |

These commands allow for full reproducibility of the simulation, analysis, optimization, and performance evaluation pipeline.


# Summary

- The simulation framework evaluates five Engression loss functions across multiple nonlinear generative models.  
- The **optimized version** adds CPU-based parallelism, reducing total runtime by **40–70%** without changing correctness.  
- Comprehensive profiling, bottleneck analysis, and regression tests ensure the optimization is safe and reliable.  
- Empirically, the **`frac` loss** consistently provides the most accurate and stable performance across all experimental setups.  
- The project is now fully reproducible, automated, and scalable for large simulation studies.

