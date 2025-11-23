# complexity_profile_optimized.py
# Run run_case_optimized() for various (REPS, N_TRAIN) combinations
# and record runtime for complexity analysis.

import os
import time

import pandas as pd
import torch

from test_optimized import run_case_optimized


# Toggle this if you want to profile with parallelization instead
USE_PARALLEL = True  # True / False
N_JOBS = 3         # or an int like 2 or 4 if USE_PARALLEL=True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Same representative scenario as in profile.py
    model_type = "preANM"
    dx, dy, k = 2, 2, 2
    true_function = "cubic"
    noise_dist = "uniform"
    noise_std = 1.0

    # Grids of REPS and N_TRAIN to test
    REPS_LIST = [1, 2, 5, 10]
    N_TRAIN_LIST = [2000, 5000, 10000]

    records = []

    for reps in REPS_LIST:
        for n_train in N_TRAIN_LIST:
            print(f"[PROFILE] REPS={reps}, N_TRAIN={n_train}, "
                  f"use_parallel={USE_PARALLEL}")

            start = time.perf_counter()
            _ = run_case_optimized(
                model_type=model_type,
                dx=dx, dy=dy, k=k,
                true_function=true_function,
                noise_dist=noise_dist,
                noise_std=noise_std,
                device=device,
                REPS=reps,
                N_TRAIN=n_train,
                use_parallel=USE_PARALLEL,
                n_jobs=N_JOBS,
            )
            elapsed = time.perf_counter() - start

            print(f"  -> time = {elapsed:.2f} sec")
            records.append(
                dict(
                    REPS=reps,
                    N_TRAIN=n_train,
                    time_sec=elapsed,
                    use_parallel=USE_PARALLEL,
                    n_jobs=N_JOBS if N_JOBS is not None else "auto",
                )
            )

    # Save results
    out_dir = "result"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "complexity_grid_optimized.csv")
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)

    print(f"[DONE] Saved timing results to {out_path}")


if __name__ == "__main__":
    main()
