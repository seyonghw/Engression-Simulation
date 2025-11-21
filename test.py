# test.py
# 72-case simulation × 10 replications each.
# Saves tidy CSV of MSEs for five engression losses under result/.

import os
import math
import itertools
import numpy as np
import pandas as pd
import torch

# --- your package modules (per your structure) ---
from engression_local.data.data_generator import (
    nonlinearity,      # used for truth surface
    generate_mats,     # A, M generators
    postanm_generator,
    preanm_generator,
)
from engression_local import engression  # training + prediction


# -----------------------
# Config
# -----------------------

MODELS = ["preANM"]
DIMS = [(2, 2, 2)]
TRUE_FUNS = ["cubic", "square"]
NOISE_DISTS = ["uniform"]
NOISE_STDS = [1.0, 2.0, 3.0]

#N_TRAIN = 10_000
#REPS = 10  # A, M draws per case
DEFAULT_N_TRAIN = 10_000
DEFAULT_REPS = 10

# Prediction config (keep consistent across losses)
PRED_TARGET = "median"
PRED_SAMPLE_SIZE = 1000

# Training config
LR = 0.005
EPOCHS = 300
BATCH = 1000


# -----------------------
# Utilities
# -----------------------
def x_support(true_function: str):
    """Return (low, high) support for X depending on h."""
    if true_function in ("cubic", "square"):
        return -2.0, 2.0
    elif true_function == "log":  # piecewise-log
        return 0.0, 5.0
    raise ValueError(f"Unknown true_function '{true_function}'")


def build_eval_grid(dx: int, low: float, high: float, device):
    """dx in {1,2}: return (G, dx) grid tensor on device."""
    if dx == 1:
        g = torch.linspace(low, high, 100, device=device).reshape(-1, 1)
        return g
    elif dx == 2:
        g = torch.linspace(low, high, 50, device=device)
        X1, X2 = torch.meshgrid(g, g, indexing="ij")
        return torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)
    else:
        raise ValueError("This experiment only uses dx in {1,2}.")


@torch.no_grad()
def true_surface_on_grid(
    x_eval: torch.Tensor,
    A: torch.Tensor,
    M: torch.Tensor,
    true_function: str
):
    """
    Compute E[Y|X=x] on the grid.
      post-ANM: M h(A x)
      pre-ANM : E_{eps_x}[ M h(A (x + eps_x)) ]  (MC)
    """
    h = nonlinearity(true_function)
    Z = x_eval @ A.T
    U = h(Z)
    return U @ M.T
    

def set_seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------
# Single-case runner
# -----------------------
def run_case(model_type, dx, dy, k, true_function, noise_dist, noise_std, device, REPS=10, N_TRAIN=10000):
    """
    Run one parameter case with REPS replications.
    Returns: list of dict rows (one per loss per replication).
    """
    rows = []

    low, high = x_support(true_function)
    x_eval = build_eval_grid(dx, low, high, device=device)  # (G, dx)

    for rep in range(REPS):
        # Seed per replication (deterministic across machines)
        set_seed_all(rep)

        # A, M for this replication
        A, M = generate_mats(dx=dx, dy=dy, k=k, seed=rep, device=device)

        # Simulate data
        gen_kwargs = dict(
            n=N_TRAIN, dx=dx, dy=dy, k=k,
            true_function=true_function,
            x_lower=low, x_upper=high,
            noise_dist=noise_dist, noise_std=noise_std,
            A=A, M=M, seed=rep, device=device,
        )
        if model_type == "postANM":
            x, y = postanm_generator(**gen_kwargs)
        else:
            x, y = preanm_generator(**gen_kwargs)

        # Ground-truth E[Y|X=x] on grid
        y_true = true_surface_on_grid(
            x_eval=x_eval, A=A, M=M,
            true_function=true_function
        )  # (G, dy)

        # Helper: train & predict for a given loss
        def fit_pred(loss_phi, beta=1):
            model = engression(
                x, y, lr=LR, loss_phi=loss_phi, beta=beta,
                num_epochs=EPOCHS, batch_size=BATCH, device=device, verbose=False
            )
            y_hat = model.predict(x_eval, target=PRED_TARGET, sample_size=PRED_SAMPLE_SIZE)
            return y_hat.reshape(-1, dy)

        # Train/predict for five losses
        y_power05 = fit_pred("power", beta=0.5)
        y_power1  = fit_pred("power", beta=1.0)
        y_exp     = fit_pred("exp")
        y_log1p   = fit_pred("log1p")
        y_frac    = fit_pred("frac")

        # MSEs
        mse = lambda yp: torch.mean((yp - y_true) ** 2).item()

        base = dict(
            model=model_type, dx=dx, dy=dy, k=k,
            h=true_function, noise_dist=noise_dist, noise_std=noise_std,
            seed_rep=rep,
        )
        rows.extend([
            dict(**base, loss_phi="power_0.5", mse=mse(y_power05)),
            dict(**base, loss_phi="power_1",   mse=mse(y_power1)),
            dict(**base, loss_phi="exp",       mse=mse(y_exp)),
            dict(**base, loss_phi="log1p",     mse=mse(y_log1p)),
            dict(**base, loss_phi="frac",      mse=mse(y_frac)),
        ])

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return rows


# -----------------------
# Main
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Expect 'result/' to already exist (per your note)
    out_dir = "result"
    if not os.path.isdir(out_dir):
        raise FileNotFoundError("Folder 'result/' not found. Please create it at repo root next to engression_local/ and examples/.")

    all_rows = []
    for model_type, (dx, dy, k), true_function, noise_dist, noise_std in itertools.product(
        MODELS, DIMS, TRUE_FUNS, NOISE_DISTS, NOISE_STDS
    ):
        print(f"[CASE] {model_type} | (dx,dy,k)=({dx},{dy},{k}) | h={true_function} | "
              f"{noise_dist}, std={noise_std}")
        rows = run_case(
            model_type=model_type, dx=dx, dy=dy, k=k,
            true_function=true_function,
            noise_dist=noise_dist, noise_std=noise_std,
            device=device,
            REPS=DEFAULT_REPS,
            N_TRAIN=DEFAULT_N_TRAIN,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(out_dir, f"mse_engression_losses.csv")
    df.to_csv(out_path, index=False)
    print(f"[DONE] Saved: {out_path}")


if __name__ == "__main__":
    main()
