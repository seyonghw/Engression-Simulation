# check_regression_mse.py
"""
Step 6: Regression tests for baseline vs optimized MSE results.

Compares:
  - result/mse_engression_losses.csv
  - result/mse_engression_losses_optimized.csv

and checks that MSE values match up to a small tolerance.
"""

import os
import sys
import numpy as np
import pandas as pd

RESULT_DIR = "result"
BASELINE_CSV = os.path.join(RESULT_DIR, "mse_engression_losses.csv")
OPT_CSV = os.path.join(RESULT_DIR, "mse_engression_losses_optimized.csv")

# Keys that should uniquely identify a row
KEY_COLS = [
    "model",
    "dx",
    "dy",
    "k",
    "h",
    "noise_dist",
    "noise_std",
    "seed_rep",
    "loss_phi",
]

# Tolerance for floating-point differences
ABS_TOL = 1e-6
REL_TOL = 1e-6


def main():
    if not os.path.exists(BASELINE_CSV):
        print(f"[ERROR] Baseline CSV not found: {BASELINE_CSV}")
        sys.exit(1)
    if not os.path.exists(OPT_CSV):
        print(f"[ERROR] Optimized CSV not found: {OPT_CSV}")
        sys.exit(1)

    base = pd.read_csv(BASELINE_CSV)
    opt = pd.read_csv(OPT_CSV)

    # Basic shape check
    if base.shape[0] != opt.shape[0]:
        print(f"[ERROR] Row count mismatch: baseline={base.shape[0]}, optimized={opt.shape[0]}")
        sys.exit(1)

    # Merge on key columns
    merged = base.merge(
        opt[KEY_COLS + ["mse"]],
        on=KEY_COLS,
        suffixes=("_base", "_opt"),
        how="outer",
        indicator=True,
    )

    # Check for any rows that don't match between the two files
    mismatched_indicator = merged["_merge"] != "both"
    if mismatched_indicator.any():
        print("[ERROR] Some rows are not matched between baseline and optimized CSVs.")
        print(merged[mismatched_indicator].head())
        sys.exit(1)

    # Compute differences
    diff = merged["mse_base"] - merged["mse_opt"]
    abs_diff = diff.abs()
    # Avoid division by zero: only compute relative diff where mse_base != 0
    rel_diff = abs_diff / (merged["mse_base"].abs() + 1e-12)

    max_abs = abs_diff.max()
    max_rel = rel_diff.max()
    mean_abs = abs_diff.mean()

    print("[INFO] Regression test: baseline vs optimized MSE")
    print(f"  Rows compared      : {len(merged)}")
    print(f"  Max abs diff       : {max_abs:.3e}")
    print(f"  Mean abs diff      : {mean_abs:.3e}")
    print(f"  Max relative diff  : {max_rel:.3e}")
    print(f"  Abs tol            : {ABS_TOL:.1e}")
    print(f"  Rel tol            : {REL_TOL:.1e}")

    # Check against tolerance
    bad_mask = (abs_diff > ABS_TOL) & (rel_diff > REL_TOL)
    if bad_mask.any():
        print("[ERROR] Regression test FAILED: some MSEs differ beyond tolerance.")
        print("Example mismatches:")
        print(
            merged.loc[bad_mask, KEY_COLS + ["mse_base", "mse_opt"]]
            .head(10)
            .to_string(index=False)
        )
        sys.exit(1)

    print("[OK] Regression test PASSED: optimized results match baseline within tolerance.")
    sys.exit(0)


if __name__ == "__main__":
    main()
