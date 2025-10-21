# plot_results.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 140})

RESULT_DIR = "result"
CSV_PATH = os.path.join(RESULT_DIR, "mse_engression_losses.csv")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

# === keep/exclude losses as you wish ===
LOSS_ORDER = ["power_0.5", "power_1", "exp", "log1p", "frac"]
LOSS_LABEL = {
    "power_0.5": "power (β=0.5)",
    "power_1":   "power (β=1)",
    "exp":       "exp",
    "log1p":     "log1p",
    "frac":      "frac",
}

# NEW: allowed dimension combos
ALLOWED_DIMS = {(1, 1, 1), (2, 2, 1), (2, 2, 2)}

def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)

def load_and_aggregate():
    df = pd.read_csv(CSV_PATH)
    needed = {"model","dx","dy","k","h","noise_dist","noise_std","seed_rep","loss_phi","mse"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")

    # filter to allowed dims only
    df = df[df[["dx","dy","k"]].apply(tuple, axis=1).isin(ALLOWED_DIMS)].copy()

    # filter to selected losses & set order
    df = df[df["loss_phi"].isin(LOSS_ORDER)].copy()
    df["loss_phi"] = pd.Categorical(df["loss_phi"], categories=LOSS_ORDER, ordered=True)

    # case-level mean MSE (avg over reps)
    case_keys = ["model","dx","dy","k","h","noise_dist","noise_std","loss_phi"]
    case_means = (
        df.groupby(case_keys, as_index=False)["mse"]
          .mean()
          .rename(columns={"mse":"mse_case_mean"})
    )
    return case_means

def plot_box_by_model_dims(case_means: pd.DataFrame):
    facet_keys = ["model","dx","dy","k"]
    for keys, block in case_means.groupby(facet_keys):
        model, dx, dy, k = keys
        # (this guard is redundant after filtering, but harmless)
        if (dx, dy, k) not in ALLOWED_DIMS:
            continue

        block = block.copy()
        block["loss_phi"] = pd.Categorical(block["loss_phi"], categories=LOSS_ORDER, ordered=True)
        block = block.sort_values("loss_phi")

        data = [block.loc[block["loss_phi"]==lp, "mse_case_mean"].values for lp in LOSS_ORDER]
        labels = [LOSS_LABEL[lp] for lp in LOSS_ORDER]
        if any(len(d)==0 for d in data):
            continue

        plt.figure(figsize=(7.2, 4.6))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("Case-level mean MSE")
        plt.title(f"Mean MSE by loss (boxes over h × noise)\nmodel={model}  |  dims=({dx},{dy},{k})")
        plt.xticks(rotation=15)
        plt.tight_layout()

        fname = f"box_case_means_{model}_dx{dx}_dy{dy}_k{k}.png"
        plt.savefig(os.path.join(FIG_DIR, fname))
        plt.close()

def main():
    ensure_dirs()
    case_means = load_and_aggregate()
    plot_box_by_model_dims(case_means)
    print(f"[DONE] wrote boxplots to {FIG_DIR}")

if __name__ == "__main__":
    main()
