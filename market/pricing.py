"""Black-Scholes pricing and Greeks for a European call.

All functions assume r = 0 
Inputs may be Python floats or tensors;
"""

import math
import torch


def _to_tensor(x):
    """Convert floats/ints to a float tensor """
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.float64)


def _d1_d2(S, K, tau, sigma):
    """d1 and d2 of the Black-Scholes formula with r = 0.

    S     : current spot price
    K     : strike
    tau   : time to maturity T - t (must be > 0)
    sigma : volatility
    """
    S, K, tau, sigma = map(_to_tensor, (S, K, tau, sigma))
    d1 = (torch.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * torch.sqrt(tau))
    d2 = d1 - (sigma * torch.sqrt(tau))
    return d1, d2


def _norm_pdf(x):
    """Standard normal density."""
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def bs_price(S, K, tau, sigma):
    """Black-Scholes call price, r = 0: C = S N(d1) - K N(d2)."""
    d1, d2 = _d1_d2(S, K, tau, sigma)
    S, K = _to_tensor(S), _to_tensor(K)
    return S * torch.special.ndtr(d1) - K * torch.special.ndtr(d2)


def bs_delta(S, K, tau, sigma):
    """Black-Scholes call delta, r = 0: N(d1)."""
    d1, _ = _d1_d2(S, K, tau, sigma)
    return torch.special.ndtr(d1)


def bs_gamma(S, K, tau, sigma):
    """Black-Scholes gamma, r = 0: N'(d1) / (S sigma sqrt(tau))."""
    d1, _ = _d1_d2(S, K, tau, sigma)
    S, tau, sigma = map(_to_tensor, (S, tau, sigma))
    return _norm_pdf(d1) / (S * sigma * torch.sqrt(tau))