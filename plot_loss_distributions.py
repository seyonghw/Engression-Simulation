# plot_loss_distributions.py
# Compare the overall MSE distributions of four engression loss functions
# (excluding 'power_1').

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"figure.dpi": 140})
RESULT_DIR = "result"
CSV_PATH = os.path.join(RESULT_DIR, "mse_engression_losses.csv")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

# include only these 4
LOSS_ORDER = ["power_0.5", "exp", "log1p", "frac"]
LOSS_LABEL = {
    "power_0.5": "power (β=0.5)",
    "exp": "exp",
    "log1p": "log1p",
    "frac": "frac",
}

def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(CSV_PATH)
    assert "loss_phi" in df and "mse" in df, "CSV missing required columns"
    # filter to selected losses
    df = df[df["loss_phi"].isin(LOSS_ORDER)].copy()
    df["loss_phi"] = pd.Categorical(df["loss_phi"], categories=LOSS_ORDER, ordered=True)
    df["loss_label"] = df["loss_phi"].map(LOSS_LABEL)
    return df

def violinplot(df):
    plt.figure(figsize=(7, 4.5))
    sns.violinplot(
        x="loss_label", y="mse", data=df,
        order=[LOSS_LABEL[l] for l in LOSS_ORDER],
        inner="box", cut=0, linewidth=1
    )
    plt.ylabel("MSE (across all cases × reps)")
    plt.xlabel("Loss function")
    plt.title("Distribution of MSE values by loss function")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "dist_violin_losses_4.png"))
    plt.close()

def densityplot(df):
    plt.figure(figsize=(7, 4.5))
    for l in LOSS_ORDER:
        sub = df[df["loss_phi"] == l]
        sns.kdeplot(sub["mse"], label=LOSS_LABEL[l], linewidth=2)
    plt.xlabel("MSE")
    plt.ylabel("Density")
    plt.title("Density of MSE across all simulation cases")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "dist_density_losses_4.png"))
    plt.close()

def ecdfplot(df):
    plt.figure(figsize=(7, 4.5))
    for l in LOSS_ORDER:
        sub = df[df["loss_phi"] == l]
        x = sub["mse"].sort_values()
        y = (np.arange(len(x)) + 1) / len(x)
        plt.plot(x, y, label=LOSS_LABEL[l], linewidth=2)
    plt.xlabel("MSE")
    plt.ylabel("Empirical CDF")
    plt.title("Cumulative distribution of MSE (ECDF)")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "dist_ecdf_losses_4.png"))
    plt.close()

def main():
    ensure_dirs()
    df = load_data()
    print(f"[INFO] Using {len(df)} rows from {CSV_PATH}")
    violinplot(df)
    densityplot(df)
    ecdfplot(df)
    print(f"[DONE] 4-loss distribution plots saved to {FIG_DIR}")

if __name__ == "__main__":
    main()
