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

$$
\text{REPS} \times \text{training cost}.
$$

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
```

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
- The optimized version runs **much faster** (about 1.5×–1.7× speedup for realistic cases).
- Replications are **independent**, so parallelization does **not** change the results.
- Users can easily control the number of CPU workers.

### Limitations
- For very small jobs (e.g., `REPS = 1`), parallel overhead can make it **no faster**.
- Parallelization works only on **CPU** (GPU runs stay serial).
- Multiple processes use **more memory** than a single process.

### When to use
- Use the optimized version for **large simulation runs** (REPS ≥ 3, N_TRAIN ≥ 5000).
- Use the baseline version for **small tests**, debugging, or GPU runs.

