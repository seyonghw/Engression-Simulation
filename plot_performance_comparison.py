# plot_performance_comparison.py
# Visualize baseline vs optimized profiling runs.
# This script should be placed at the repo root (NOT inside result/).

import os
import pandas as pd
import matplotlib.pyplot as plt

# Input files (inside result/)
RESULT_DIR = "result"
BASELINE_CSV = os.path.join(RESULT_DIR, "complexity_grid.csv")
OPTIMIZED_CSV = os.path.join(RESULT_DIR, "complexity_grid_optimized.csv")

# Output figures
FIG_DIR = os.path.join(RESULT_DIR, "figures")


def load_and_merge():
    """Load and merge baseline + optimized profiling results."""
    base = pd.read_csv(BASELINE_CSV)
    opt = pd.read_csv(OPTIMIZED_CSV)

    merged = base.merge(
        opt[["REPS", "N_TRAIN", "time_sec"]],
        on=["REPS", "N_TRAIN"],
        suffixes=("_base", "_opt"),
    )
    merged["speedup"] = merged["time_sec_base"] / merged["time_sec_opt"]

    # Optional pretty rounding
    merged["time_sec_base"] = merged["time_sec_base"].round(2)
    merged["time_sec_opt"] = merged["time_sec_opt"].round(2)
    merged["speedup"] = merged["speedup"].round(2)

    return merged


def plot_runtime_scatter(df):
    """Scatter plot: Baseline vs Optimized runtime."""
    os.makedirs(FIG_DIR, exist_ok=True)

    max_time = max(df["time_sec_base"].max(), df["time_sec_opt"].max())

    plt.figure(figsize=(6, 6))
    plt.plot([0, max_time], [0, max_time], color="gray", linestyle="--")

    plt.scatter(df["time_sec_base"], df["time_sec_opt"], color="blue")

    for _, row in df.iterrows():
        label = f"R{int(row['REPS'])}-N{int(row['N_TRAIN'])}"
        plt.annotate(label, (row["time_sec_base"], row["time_sec_opt"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=8)

    plt.xlabel("Baseline time (sec)")
    plt.ylabel("Optimized time (sec)")
    plt.title("Baseline vs Optimized Runtime")
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "runtime_scatter.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def plot_speedup_bar(df):
    """Bar plot: Speedup = baseline / optimized."""
    os.makedirs(FIG_DIR, exist_ok=True)

    df_sorted = df.sort_values(["REPS", "N_TRAIN"]).copy()
    labels = [f"R{int(r.REPS)}-N{int(r.N_TRAIN)}" for _, r in df_sorted.iterrows()]

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(df_sorted)), df_sorted["speedup"], color="green")
    plt.axhline(1.0, linestyle="--", color="gray")

    plt.xticks(range(len(df_sorted)), labels, rotation=45, ha="right")
    plt.ylabel("Speedup (baseline / optimized)")
    plt.title("Speedup Across (REPS, N_TRAIN)")
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "speedup_bar.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def main():
    if not os.path.exists(BASELINE_CSV):
        raise FileNotFoundError(f"Missing {BASELINE_CSV}")
    if not os.path.exists(OPTIMIZED_CSV):
        raise FileNotFoundError(f"Missing {OPTIMIZED_CSV}")

    df = load_and_merge()
    print("[INFO] Loaded profiling results:")
    print(df)

    plot_runtime_scatter(df)
    plot_speedup_bar(df)


if __name__ == "__main__":
    main()
