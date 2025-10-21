import torch
from .utils_local import vectorize
from torch.linalg import vector_norm


def _phi_on_sqdist(d2, phi="power", beta=0.5, eps=0.0):
    """
    Apply phi to squared distances.

    Args:
        d2 (Tensor): squared distances >= 0
        phi (str): one of ["power", "exp", "log1p", "frac"]
        beta (float): used only when phi == "power"
        eps (float): small positive number to stabilize fractional powers (only for "power")
    """
    if phi == "power":              # φ(x) = x^beta  on x = ||·||^2
        return (d2 + eps).pow(beta)
    elif phi == "exp":              # φ(x) = 1 - exp(-x/2)
        return 1.0 - torch.exp(-0.5 * d2)
    elif phi == "log1p":            # φ(x) = log(1 + x)
        return torch.log1p(d2)
    elif phi == "frac":       # φ(x) = x / (1 + x)
        return d2 / (1.0 + d2)
    else:
        raise ValueError(f"Unknown phi '{phi}'")

def energy_loss(x_true, x_est, phi="power", beta=0.5, verbose=True):
    """Loss function based on the energy score.

    Args:
        x_true (torch.Tensor): iid samples from the true distribution of shape (data_size, data_dim)
        x_est (list of torch.Tensor): 
            - a list of length sample_size, where each element is a tensor of shape (data_size, data_dim) that contains one sample for each data point from the estimated distribution, or 
            - a tensor of shape (data_size*sample_size, response_dim) such that x_est[data_size*(i-1):data_size*i,:] contains one sample for each data point, for i = 1, ..., sample_size.
        phi (str): one of ["power","exp","log1p","frac"]
        beta (float): power parameter in the energy score.
        verbose (bool): whether to return two terms of the loss.

    Returns:
        loss (torch.Tensor): energy loss.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x_true = vectorize(x_true).unsqueeze(1)
    if not isinstance(x_est, list):
        x_est = list(torch.split(x_est, x_true.shape[0], dim=0))
    m = len(x_est)
    x_est = [vectorize(x_est[i]).unsqueeze(1) for i in range(m)]
    x_est = torch.cat(x_est, dim=1)

    d2_1 = (x_est - x_true).pow(2).sum(dim=2)         
    s1 = _phi_on_sqdist(d2_1, phi=phi, beta=beta, eps=EPS).mean()
    d_kk = torch.cdist(x_est, x_est, p=2)         
    d2_kk = d_kk.pow(2)
    s2 = _phi_on_sqdist(d2_kk, phi=phi, beta=beta, eps=EPS).mean() * m / (m - 1)
        
    if verbose:
        return torch.cat([(s1 - s2 / 2).reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return (s1 - s2 / 2)
    

def energy_loss_two_sample(x0, x, xp, x0p=None, phi="power", beta=0.5, verbose=True, weights=None):
    """Loss function based on the energy score (estimated based on two samples).
    
    Args:
        x0 (torch.Tensor): an iid sample from the true distribution.
        x (torch.Tensor): an iid sample from the estimated distribution.
        xp (torch.Tensor): another iid sample from the estimated distribution.
        xp0 (torch.Tensor): another iid sample from the true distribution.
        phi (str): one of ["power","exp","log1p","frac"]
        beta (float): power parameter in the energy score.
        verbose (bool):  whether to return two terms of the loss.
    
    Returns:
        loss (torch.Tensor): energy loss.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)
    if weights is None:
        weights = 1 / x0.size(0)


    d2_x_x0  = (x  - x0 ).pow(2).sum(dim=1)
    d2_xp_x0 = (xp - x0 ).pow(2).sum(dim=1)
    s1 = (_phi_on_sqdist(d2_x_x0,  phi=phi, beta=beta, eps=EPS) * weights).sum() / 2.0 + (_phi_on_sqdist(d2_xp_x0, phi=phi, beta=beta, eps=EPS) * weights).sum() / 2.0

    d2_x_xp  = (x  - xp ).pow(2).sum(dim=1)
    s2 = (_phi_on_sqdist(d2_x_xp,  phi=phi, beta=beta, eps=EPS) * weights).sum()

    if x0p is None:
        loss = s1 - s2/2.0
        if verbose:
            return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
        else:
            return loss
    else:
        x0p = vectorize(x0p)
        d2_x_x0p  = (x  - x0p).pow(2).sum(dim=1)
        d2_xp_x0p = (xp - x0p).pow(2).sum(dim=1)
        s1 = (_phi_on_sqdist(d2_x_x0,   phi=phi, beta=beta, eps=EPS).sum() +
              _phi_on_sqdist(d2_xp_x0,  phi=phi, beta=beta, eps=EPS).sum() +
              _phi_on_sqdist(d2_x_x0p,  phi=phi, beta=beta, eps=EPS).sum() +
              _phi_on_sqdist(d2_xp_x0p, phi=phi, beta=beta, eps=EPS).sum()) / 4.0
        s2 = _phi_on_sqdist(d2_x_xp,  phi=phi, beta=beta, eps=EPS).sum()
        d2_x0_x0p = (x0 - x0p).pow(2).sum(dim=1)
        s3 = _phi_on_sqdist(d2_x0_x0p, phi=phi, beta=beta, eps=EPS).sum()
        loss = s1 - s2/2.0 - s3/2.0
        if verbose:
            return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
        else:
            return loss

