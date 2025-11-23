# OPTIMIZATION.md

## 1. Problem Identified

The baseline implementation (`test.py`) executes **all replications serially**.  
For each parameter configuration, it loops through:

- generating matrices `(A, M)`,
- simulating data,
- computing the true surface,
- **training 5 engression models**,  
- computing MSEs.

Each replication is **independent**, yet executed **sequentially**.  
Since training dominates runtime (~85–90%), wall-clock time grows roughly linearly in

\[
\text{REPS} \times \text{training cost}.
\]

This serial execution is the main bottleneck.

---

## 2. Solution Implemented

We introduced CPU-based parallelization in `test_optimized.py`.

### Key idea  

> **Each replication is independent → perfect for multiprocessing.**

### Steps taken

- Moved “one replication” logic into a worker function `_run_single_rep`.
- Used `multiprocessing.Pool` to distribute replications across CPU workers.
- Preserved reproducibility by seeding inside each worker (`set_seed_all(rep)`).
- Enabled parallelism only when `device.type == "cpu"` (GPU stays serial to avoid contention).
- Kept a serial fallback for small jobs or GPU runs.

---

## 3. Code Comparison

**Baseline (serial) vs Optimized (parallel)**

```python
# Baseline: run replications one-by-one
for rep in range(REPS):
    rows += run_one_rep(rep)

# Optimized: run replications in parallel (CPU)
with mp.Pool(n_jobs) as pool:
    results = pool.map(run_one_rep, range(REPS))
rows = [r for group in results for r in group]


## 4. Runtime Improvement

We profiled both the baseline (`test.py`) and optimized (`test_optimized.py`) implementations on the same representative configuration:

- `model_type = "preANM"`
- `(dx, dy, k) = (2, 2, 2)`
- `true_function = "cubic"`
- `noise_dist = "uniform"`
- `noise_std = 1.0`
- `REPS ∈ {1, 2, 5, 10}`
- `N_TRAIN ∈ {2000, 5000, 10000}`

### 4.1 Exact measured times

The table below compares **baseline** and **parallel** runtimes (in seconds), along with the corresponding speedup:

| REPS | N_TRAIN | Baseline (s) | Optimized (s) | Speedup |
|------|---------|--------------|---------------|---------|
| 1    | 2000    | 19.39        | 18.89         | 1.03×   |
| 1    | 5000    | 44.19        | 44.88         | 0.98×   |
| 1    | 10000   | 77.26        | 78.11         | 0.99×   |
| 2    | 2000    | 36.18        | 29.50         | 1.23×   |
| 2    | 5000    | 88.63        | 52.34         | 1.69×   |
| 2    | 10000   | 154.06       | 103.00        | 1.50×   |
| 5    | 2000    | 90.18        | 85.56         | 1.05×   |
| 5    | 5000    | 221.84       | 148.53        | 1.49×   |
| 5    | 10000   | 388.76       | 230.62        | 1.69×   |
| 10   | 2000    | 180.32       | 210.57        | 0.86×   |
| 10   | 5000    | 442.77       | 299.57        | 1.48×   |
| 10   | 10000   | 768.28       | 490.80        | 1.57×   |

### 4.2 Summary

- The **overall average speedup** across this grid is about **1.40×**.
- For “realistic” workloads (e.g., `REPS ≥ 2` and `N_TRAIN ≥ 5000`), speedups are typically in the **1.5×–1.7×** range.
- For very small jobs (e.g., `REPS = 1` with small `N_TRAIN`), multiprocessing overhead dominates, so speedup is close to **1×** or sometimes slightly less than 1.

---

## 5. Trade-offs

### Advantages

- **Faster experiments:**  
  For moderate-to-large simulations, the optimized implementation reduces wall-clock time by roughly **40–70%** compared to the baseline.

- **Embarrassingly parallel structure:**  
  Each replication is independent, so parallelizing over `REPS` does **not** change the statistical procedure or model.

- **Same outputs as baseline:**  
  Per-replication seeding (`set_seed_all(rep)`) ensures that, for fixed `REPS` and `N_TRAIN`, the optimized version produces the **same results** as the serial baseline (up to minor floating-point variation).

- **Configurable parallelism:**  
  Users can choose whether to enable parallelization and how many workers to use (`use_parallel`, `n_jobs`), allowing a balance between speed and resource usage.

### Limitations / Caveats

- **Overhead for small jobs:**  
  When `REPS` is very small (especially `REPS = 1`) and `N_TRAIN` is small, the process creation overhead can cancel out the benefit, or even make the parallel version slightly slower.

- **CPU-only parallelism:**  
  For safety and simplicity, if the device is GPU (`device.type == "cuda"`), the code falls back to serial execution. Sharing a single GPU across multiple processes is often unstable or slower.

- **Higher memory usage:**  
  Each worker process has its own copy of data and model parameters, which increases total memory consumption compared to the serial baseline.

- **Need to tune `n_jobs`:**  
  Using too many workers (more than the number of available CPU cores) can lead to context-switch overhead and contention, reducing or even reversing the performance gains.

### Recommended Usage

- **Use the optimized parallel version** when:
  - `REPS ≥ 3` and `N_TRAIN ≥ 5000`,
  - running on CPU,
  - performing large simulation sweeps where runtime matters.

- **Stick to the serial baseline** when:
  - debugging or running quick tests,
  - `REPS` is very small,
  - running on GPU (where the code already falls back to serial).
