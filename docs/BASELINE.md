Here is the **BASELINE.md** you requested, following **exactly the structure you specified**, and **using the computed results** from your uploaded `complexity_grid.csv`.

You can copy this into:

```
docs/BASELINE.md
```

---

# **BASELINE.md**

## 1. Runtime Profiling

We evaluated baseline runtime using a full grid of:

* **REPS ∈ {1, 2, 5, 10}**
* **N_TRAIN ∈ {2000, 5000, 10000}**

For each pair ((\text{REPS}, N_{\text{TRAIN}})), the full set of reduced scenarios was executed:

* Models: `preANM`
* DIMS: `(2, 2, 2)`
* True functions: `cubic`, `square`
* Noise std: `1.0, 2.0, 3.0`

The measured total runtimes (in seconds) are:

### **Table 1. Baseline Runtime Profiling Results**

| REPS | N_TRAIN | Runtime (sec) |
| ---- | ------- | ------------- |
| 1    | 2000    | 19.39         |
| 1    | 5000    | 44.19         |
| 1    | 10000   | 77.26         |
| 2    | 2000    | 36.18         |
| 2    | 5000    | 88.63         |
| 2    | 10000   | 154.06        |
| 5    | 2000    | 90.18         |
| 5    | 5000    | 221.84        |
| 5    | 10000   | 388.76        |
| 10   | 2000    | 180.32        |
| 10   | 5000    | 442.77        |
| 10   | 10000   | 768.28        |

These measurements form the empirical basis for computational complexity analysis.

---

## 2. Computational Complexity

We analyze complexity along two separate axes:

* **(1) Runtime vs REPS** (fixing N_TRAIN)
* **(2) Runtime vs N_TRAIN** (fixing REPS)

---

### **2.1 Complexity: REPS vs Runtime**

Holding N_TRAIN fixed, runtime is nearly **linear in REPS**.

Example for N_TRAIN = 2000:

| REPS | Runtime  |
| ---- | -------- |
| 1    | 19.39 s  |
| 2    | 36.18 s  |
| 5    | 90.18 s  |
| 10   | 180.32 s |

Doubling REPS approximately doubles runtime:

[
\text{time} \propto \text{REPS}^{,1.0}.
]

**Conclusion:**

> Runtime scales **linearly** with REPS because each replication performs an independent full training of 5 engression models.

---

### **2.2 Complexity: N_TRAIN vs Runtime**

Holding REPS fixed, runtime increases approximately **linearly** in N_TRAIN.

Example for REPS = 1:

| N_TRAIN | Runtime |
| ------- | ------- |
| 2000    | 19.39 s |
| 5000    | 44.19 s |
| 10000   | 77.26 s |

Example for REPS = 10:

| N_TRAIN | Runtime  |
| ------- | -------- |
| 2000    | 180.32 s |
| 5000    | 442.77 s |
| 10000   | 768.28 s |

The growth is consistent with:

[
\text{time} \approx c \cdot N_{\text{TRAIN}}^{,1.0}.
]

**Explanation:**
The training loop processes the data in minibatches (`batch_size=1000`), so the number of iterations per epoch scales linearly with `N_TRAIN`.

**Conclusion:**

> Runtime grows **approximately linearly** in N_TRAIN.

