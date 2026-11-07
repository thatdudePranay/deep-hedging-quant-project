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

# ---------------------------------------------------------------------------
# Heston (semi-closed form via Fourier inversion), r = 0
# ---------------------------------------------------------------------------
# Uses numpy/scipy rather than torch: it returns a single price (e.g. the premium p0),
# so it does not need batching or autograd.

import numpy as np
from scipy.integrate import quad


def heston_cf(u, T, v0, kappa, theta, xi, rho):
    """Characteristic function E[exp(i u log(S_T / S_0))] under Heston with r = 0.

    Uses the "little trap" form (Albrecher et al. 2007), which avoids branch-cut problems
    of the complex logarithm. Parameters in common notation:
        kappa = mean-reversion speed (alpha in market/simulators.py)
        theta = long-run variance    (b)
        xi    = vol of variance      (sigma)
        rho   = spot/variance correlation, v0 = initial variance
    u may be complex.
    """
    beta = kappa - rho * xi * 1j * u
    d = np.sqrt(beta**2 + xi**2 * (1j * u + u**2))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((beta - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (beta - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def heston_price(K, T, params=None, upper=200.0):
    """European call price under Heston, r = 0, via Lewis' single-integral formula:

        C = S0 - sqrt(S0 K) / pi * int_0^inf Re[ e^{i u log(S0/K)} phi(u - i/2) ] / (u^2 + 1/4) du

    params: a HestonParams from market.simulators (the same object used to simulate the paths),
            so pricing and simulation always share one set of parameters. Defaults to HestonParams().
    Returns a Python float.

    Checks: phi(0) = 1, and as params.sigma -> 0 with v0 = b the price converges to the
    BS price with volatility sqrt(v0).
    """
    from market.simulators import HestonParams
    p = HestonParams() if params is None else params
    S0 = p.s0
    k = np.log(S0 / K)

    def integrand(u):
        phi = heston_cf(u - 0.5j, T, v0=p.v0, kappa=p.alpha, theta=p.b, xi=p.sigma, rho=p.rho)
        return np.real(np.exp(1j * u * k) * phi) / (u**2 + 0.25)

    integral, _ = quad(integrand, 0.0, upper, limit=500)
    return float(S0 - np.sqrt(S0 * K) / np.pi * integral)
