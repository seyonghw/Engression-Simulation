# bottleneck.py
# Measure runtimes of key steps inside run_case() to find bottlenecks.

import os
import time

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
# Config (restricted case)
# -----------------------

MODEL_TYPE = "preANM"
DX, DY, K = 2, 2, 2
TRUE_FUNCTION = "cubic"
NOISE_DIST = "uniform"
NOISE_STD = 3.0

DEFAULT_N_TRAIN = 10_000
DEFAULT_REPS = 10  # number of replications to profile

# Prediction config (keep consistent across losses)
PRED_TARGET = "median"
PRED_SAMPLE_SIZE = 1000

# Training config
LR = 0.005
EPOCHS = 300
BATCH = 1000


# -----------------------
# Utilities (copied from test.py)
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
    (Here we use the deterministic post-ANM form as ground truth.)
    """
    h = nonlinearity(true_function)
    Z = x_eval @ A.T
    U = h(Z)
    return U @ M.T


def set_seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------
# Bottleneck runner
# -----------------------
def run_bottleneck(
    model_type=MODEL_TYPE,
    dx=DX,
    dy=DY,
    k=K,
    true_function=TRUE_FUNCTION,
    noise_dist=NOISE_DIST,
    noise_std=NOISE_STD,
    device=None,
    REPS=DEFAULT_REPS,
    N_TRAIN=DEFAULT_N_TRAIN,
):
    """
    Run a restricted case multiple times and record runtimes of
      1) generate_mats
      2) true_surface_on_grid
      3) engression (training)
      4) model.predict (prediction)

    Returns
    -------
    records : list of dict
        Each row corresponds to (rep, loss_phi) and includes timing info.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    low, high = x_support(true_function)
    x_eval = build_eval_grid(dx, low, high, device=device)  # (G, dx)

    records = []

    for rep in range(REPS):
        print(f"[BOTTLENECK] rep={rep}")
        set_seed_all(rep)

        # 1) generate_mats
        t0 = time.perf_counter()
        A, M = generate_mats(dx=dx, dy=dy, k=k, seed=rep, device=device)
        t1 = time.perf_counter()
        time_gen_mats = t1 - t0

        # (extra) simulate data (x, y) — often non-trivial, so we time it too
        gen_kwargs = dict(
            n=N_TRAIN, dx=dx, dy=dy, k=k,
            true_function=true_function,
            x_lower=low, x_upper=high,
            noise_dist=noise_dist, noise_std=noise_std,
            A=A, M=M, seed=rep, device=device,
        )
        t0 = time.perf_counter()
        if model_type == "postANM":
            x, y = postanm_generator(**gen_kwargs)
        else:
            x, y = preanm_generator(**gen_kwargs)
        t1 = time.perf_counter()
        time_data_gen = t1 - t0

        # 2) true_surface_on_grid
        t0 = time.perf_counter()
        y_true = true_surface_on_grid(
            x_eval=x_eval, A=A, M=M,
            true_function=true_function
        )  # (G, dy)
        t1 = time.perf_counter()
        time_true_surface = t1 - t0

        dy_local = y_true.shape[1]

        # Helper: train & predict for a given loss with timing
        def fit_pred_timed(loss_phi, beta=1.0):
            # 3) engression (training)
            t_train_start = time.perf_counter()
            model = engression(
                x, y,
                lr=LR,
                loss_phi=loss_phi,
                beta=beta,
                num_epochs=EPOCHS,
                batch_size=BATCH,
                device=device,
                verbose=False,
            )
            t_train_end = time.perf_counter()
            time_train = t_train_end - t_train_start

            # 4) model.predict
            t_pred_start = time.perf_counter()
            y_hat = model.predict(
                x_eval,
                target=PRED_TARGET,
                sample_size=PRED_SAMPLE_SIZE,
            )
            t_pred_end = time.perf_counter()
            time_predict = t_pred_end - t_pred_start

            y_hat = y_hat.reshape(-1, dy_local)
            return y_hat, time_train, time_predict

        # Train/predict for five losses (like test.py)
        configs = [
            ("power_0.5", "power", 0.5),
            ("power_1",   "power", 1.0),
            ("exp",       "exp",   1.0),
            ("log1p",     "log1p", 1.0),
            ("frac",      "frac",  1.0),
        ]

        mse = lambda yp: torch.mean((yp - y_true) ** 2).item()

        for loss_label, loss_phi, beta in configs:
            y_hat, time_train, time_predict = fit_pred_timed(loss_phi, beta=beta)
            mse_val = mse(y_hat)

            rec = dict(
                model=model_type,
                dx=dx,
                dy=dy,
                k=k,
                h=true_function,
                noise_dist=noise_dist,
                noise_std=noise_std,
                seed_rep=rep,
                loss_label=loss_label,
                loss_phi=loss_phi,
                beta=beta,
                mse=mse_val,
                time_gen_mats=time_gen_mats,
                time_data_gen=time_data_gen,
                time_true_surface=time_true_surface,
                time_train=time_train,
                time_predict=time_predict,
            )
            records.append(rec)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return records


# -----------------------
# Main: run and save CSV
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    records = run_bottleneck(device=device)

    out_dir = "result"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bottleneck_times.csv")

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)

    print(f"[DONE] Saved bottleneck timings to {out_path}")


if __name__ == "__main__":
    main()
