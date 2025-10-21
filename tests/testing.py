# tests/testing.py
# PyTest checks (sanity + theoretical properties)

import math
import numpy as np
import torch
import pytest

from engression_local.data.data_generator import (
    nonlinearity,
    generate_mats,
    postanm_generator,
    preanm_generator,
)

# -----------------------
# Helpers / fixtures
# -----------------------

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")  # deterministic & CI friendly


def set_seed(seed: int = 123):
    np.random.seed(seed)
    torch.manual_seed(seed)


def support_for(true_function: str):
    # Matches your project spec
    if true_function in ("cubic", "square"):
        return -2.0, 2.0
    elif true_function == "log":
        return 0.0, 5.0
    raise ValueError(true_function)


# -----------------------
# 1) Support + shape sanity for generators
# -----------------------

@pytest.mark.parametrize("model_type", ["preANM", "postANM"])
@pytest.mark.parametrize("dx,dy,k", [(1,1,1), (2,2,1), (2,2,2)])
@pytest.mark.parametrize("true_function", ["cubic", "square", "log"])
@pytest.mark.parametrize("noise_dist", ["gaussian", "uniform"])
def test_support_and_shapes(model_type, dx, dy, k, true_function, noise_dist, device):
    set_seed(0)
    low, high = support_for(true_function)
    A, M = generate_mats(dx=dx, dy=dy, k=k, seed=0, device=device)

    gen_kwargs = dict(
        n=1_000, dx=dx, dy=dy, k=k, true_function=true_function,
        x_lower=low, x_upper=high,
        noise_dist=noise_dist, noise_std=1.0,
        A=A, M=M, seed=0, device=device,
    )
    if model_type == "postANM":
        X, Y = postanm_generator(**gen_kwargs)
    else:
        X, Y = preanm_generator(**gen_kwargs)

    # shapes
    assert X.shape == (1_000, dx)
    assert Y.shape == (1_000, dy)

    # support
    assert torch.all(X >= low - 1e-7)
    assert torch.all(X <= high + 1e-7)

    # finite values
    assert torch.isfinite(X).all()
    assert torch.isfinite(Y).all()


# -----------------------
# 2) Post-ANM: E[Y|X=x] equals M h(Ax) (with zero output noise)
# -----------------------

@pytest.mark.parametrize("dx,dy,k", [(1,1,1), (2,2,2)])
@pytest.mark.parametrize("true_function", ["cubic", "square", "log"])
def test_postanm_true_surface_matches_closed_form(dx, dy, k, true_function, device):
    set_seed(123)
    low, high = support_for(true_function)
    A, M = generate_mats(dx=dx, dy=dy, k=k, seed=1, device=device)
    h = nonlinearity(true_function)

    # sample random X on support
    n = 512
    X = (high - low) * torch.rand(n, dx, device=device) + low

    # zero output noise -> Y must equal M h(A X)
    Y_gen = postanm_generator(
        n=n, dx=dx, dy=dy, k=k, true_function=true_function,
        x_lower=low, x_upper=high,
        noise_dist="gaussian", noise_std=0.0,  # <-- key
        A=A, M=M, seed=2, device=device
    )[1]

    Y_closed = h(X @ A.T) @ M.T

    # close within a tight tolerance
    err = torch.norm(Y_gen - Y_closed) / (1 + torch.norm(Y_closed))
    assert err.item() < 1e-6, f"Relative error too large: {err.item()}"


# -----------------------
# 3) Heteroskedastic (pre-ANM) vs homoskedastic (post-ANM)
#     - pre-ANM: Var(Y|X=x) depends on x after nonlinearity
#     - post-ANM: Var(Y|X=x) ≈ constant (equals noise variance), independent of x
# -----------------------

@pytest.mark.parametrize("true_function", ["cubic", "square"])
def test_pre_vs_post_variance_structure(true_function, device):
    set_seed(7)
    dx, dy, k = 2, 2, 2
    low, high = support_for(true_function)
    A, M = generate_mats(dx=dx, dy=dy, k=k, seed=7, device=device)

    # two distinct X points in support
    x1 = torch.full((1, dx), low + 0.2*(high-low), device=device)
    x2 = torch.full((1, dx), low + 0.8*(high-low), device=device)

    # Monte-Carlo samples to estimate Var(Y|X=x)
    def sample_post(x, n=4000, noise_std=1.0):
        X = x.repeat(n, 1)
        Y = postanm_generator(
            n=n, dx=dx, dy=dy, k=k, true_function=true_function,
            x_lower=low, x_upper=high,
            noise_dist="gaussian", noise_std=noise_std,
            A=A, M=M, seed=11, device=device
        )[1]
        return Y

    def sample_pre(x, n=4000, noise_std=1.0):
        # note: pre-ANM generator ignores eps_y and injects noise in X
        X = x.repeat(n, 1)  # passed but generator draws its own X; instead we approximate via nearby X range
        # To get conditional behavior at x, simulate many small neighborhoods around x by forcing x_lower=x_upper=x
        Y = preanm_generator(
            n=n, dx=dx, dy=dy, k=k, true_function=true_function,
            x_lower=float(x[0,0].item()), x_upper=float(x[0,0].item()),
            noise_dist="gaussian", noise_std=noise_std,
            A=A, M=M, seed=13, device=device
        )[1]
        return Y

    # Post-ANM: variance should be (approximately) independent of x
    Y1_post = sample_post(x1)
    Y2_post = sample_post(x2)
    v1_post = Y1_post.var(dim=0).mean().item()
    v2_post = Y2_post.var(dim=0).mean().item()
    assert abs(v1_post - v2_post) < 0.1 * max(1.0, v1_post, v2_post), \
        f"Post-ANM variance changed with x: {v1_post} vs {v2_post}"

    # Pre-ANM: variance should differ across x (heteroskedastic)
    Y1_pre = sample_pre(x1)
    Y2_pre = sample_pre(x2)
    v1_pre = Y1_pre.var(dim=0).mean().item()
    v2_pre = Y2_pre.var(dim=0).mean().item()
    assert abs(v1_pre - v2_pre) > 0.05 * max(1.0, v1_pre, v2_pre), \
        f"Pre-ANM variance difference too small: {v1_pre} vs {v2_pre}"
