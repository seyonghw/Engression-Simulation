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

$$
\text{time} \propto \text{REPS}^{,1.0}.
$$

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

$$
\text{time} \approx c \cdot N_{\text{TRAIN}}^{,1.0}.
$$

**Explanation:**
The training loop processes the data in minibatches (`batch_size=1000`), so the number of iterations per epoch scales linearly with `N_TRAIN`.

**Conclusion:**

> Runtime grows **approximately linearly** in N_TRAIN.


## 3. Bottleneck Analysis

To see which part of the pipeline is slowest, we timed four main steps inside `run_case()`:

1. `generate_mats` (matrix generation)
2. data generation (`preanm_generator`)
3. `true_surface_on_grid` (truth on a grid)
4. `engression` training
5. `model.predict` (prediction on a grid with MC samples)

Using **REPS = 10**, **N_TRAIN = 10,000**, and 5 losses per replication, we obtained the following **average times per model**:

| Step                  | Avg time (sec) | Comment       |
|----------------------|----------------|---------------|
| Matrix generation    | ~0.006         | negligible    |
| Data generation      | ~0.007         | negligible    |
| True surface         | ~0.0003        | negligible    |
| **Training**         | **~13.2**      | **dominant**  |
| Prediction           | ~2.4           | secondary     |

### Main takeaway

> **Training is the clear bottleneck.**  
> Almost all runtime comes from the `engression` training loop; matrix generation, data simulation, and true-surface computation are essentially free in comparison. Prediction is the second-largest cost but still much smaller than training.

In practice, any attempt to speed up the baseline should focus on:

- reducing the number of epochs,
- reducing the number of replications,
- or simplifying / shrinking the network used in `engression`.

