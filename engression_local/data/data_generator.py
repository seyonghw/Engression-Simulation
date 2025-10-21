import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def nonlinearity(true_function="softplus"):
    if true_function == "softplus":
        return lambda z: F.softplus(z)
    elif true_function == "cubic":
        return lambda z: (z ** 3) / 3.0
    elif true_function == "square":  # (ReLU(z))^2 / 2
        return lambda z: (F.relu(z) ** 2) / 2.0
    elif true_function == "log":     # piecewise from your 1d version, applied element-wise
        # for numerical stability use torch.where
        return lambda z: torch.where(
            z <= 2.0, z/3.0 + torch.log(torch.tensor(3.0, device=z.device)) - 2.0/3.0,
            torch.log(1.0 + z)
        )
    else:
        raise ValueError(f"Unknown nonlinearity '{true_function}'")
     
def generate_X(n=10000, dx=1, x_lower=0, x_upper=5, device=None):
    return (x_upper - x_lower) * torch.rand(n, dx, device=device) + x_lower

def generate_eps(n=10000, d=1, noise_dist = "gaussian", noise_std=1, device=None):
    if noise_dist == "gaussian":
        return torch.randn(n, d, device=device) * noise_std
    elif noise_dist == "uniform":
        return (torch.rand(n, 1) - 0.5)*noise_std*np.sqrt(12)
    else:
        raise ValueError(f"Unknown epsilon distribution '{noise_dist}'")
    
def generate_mats(dx=1, dy=1, k=1, seed=None, device=None):
    """
    Make default linear maps:
      A: (k x dx), M: (dy x k)
    Using i.i.d. N(0,1) with fan-in scaling.
    """
    torch.manual_seed(seed)
    A = torch.randn(k, dx, device=device) / (dx ** 0.5)
    M = torch.randn(dy, k, device=device) / (k ** 0.5)
    return A, M
    
# -------------------------
# POST-ANM: Y = M h(A X) + eps_y
# -------------------------

def postanm_generator(
    n = 10_000,
    dx = 1,
    dy = 1,
    k = 1,
    true_function = "softplus",
    x_lower=0,
    x_upper=5,
    noise_dist = "gaussian",    # 'gaussian' or 'uniform'
    noise_std=1,
    A = None,
    M = None,
    seed = None,
    device = None
):
    """
    Multivariate post-ANM simulator: Y = M h(A X) + eps_y.

    Returns:
        X: (n, dx)
        Y: (n, dy)
        (optional) A: (k, dx), M: (dy, k)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # inputs and noise
    X = generate_X(n, dx, x_lower, x_upper, device)
    eps = generate_eps(n, dy, noise_dist, noise_std, device)

    # nonlinearity, element-wise
    h = nonlinearity(true_function)

    # compute
    Z = X @ A.T              # (n, k)
    U = h(Z)                 # (n, k) element-wise
    Y = U @ M.T + eps     # (n, dy)

    return X, Y

# -------------------------
# PRE-ANM: Y = M h(A (X + eps_x))
# -------------------------

def preanm_generator(
    n = 10_000,
    dx = 1,
    dy = 1,
    k = 1,
    true_function = "softplus",
    x_lower=0,
    x_upper=5,
    noise_dist = "gaussian",    # 'gaussian' or 'uniform'
    noise_std=1,
    A = None,
    M = None,
    seed = None,
    device = None
):
    """
    Multivariate pre-ANM simulator: Y = M h(A (X + eps_x)).

    Returns:
        X: (n, dx)           # the underlying covariate sample
        Y: (n, dy)           # outcome
        (optional) A: (k, dx), M: (dy, k)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # inputs and noise
    X = generate_X(n, dx, x_lower, x_upper, device)
    eps_x = generate_eps(n, dx, noise_dist, noise_std, device)

    # nonlinearity, element-wise
    h = nonlinearity(true_function)

    # compute
    Z = (X + eps_x) @ A.T    # (n, k)
    U = h(Z)                 # (n, k)
    Y = U @ M.T              # (n, dy)  # no output noise in pre-ANM per your spec

    return X, Y