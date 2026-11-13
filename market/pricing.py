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
from market.simulators import HestonParams


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


def heston_price(K, T, params=None, upper=None):
    """European call price under Heston, r = 0, via Lewis' single-integral formula:

        C = S0 - sqrt(S0 K) / pi * int_0^inf Re[ e^{i u log(S0/K)} phi(u - i/2) ] / (u^2 + 1/4) du

    params: a HestonParams from market.simulators (the same object used to simulate the paths),
            so pricing and simulation always share one set of parameters. Defaults to HestonParams().
    Returns a Python float.

    Checks: phi(0) = 1, and as params.sigma -> 0 with v0 = b the price converges to the
    BS price with volatility sqrt(v0).
    """

    p = HestonParams() if params is None else params
    S0 = p.s0
    k = np.log(S0 / K)
    if upper is None:
        upper = min(60.0 / math.sqrt(T * p.b), 5000.0)

    def integrand(u):
        phi = heston_cf(u - 0.5j, T, v0=p.v0, kappa=p.alpha, theta=p.b, xi=p.sigma, rho=p.rho)
        return np.real(np.exp(1j * u * k) * phi) / (u**2 + 0.25)

    integral, _ = quad(integrand, 0.0, upper, limit=2000)
    return float(S0 - np.sqrt(S0 * K) / np.pi * integral)


def heston_greeks(S, v, tau, K, params=None, N=512):
    """Vectorized Heston call price, delta and vega (r = 0) for a whole batch of states.

    Same Lewis formula as `heston_price`,

        C = S - sqrt(SK)/pi * I,    I = int_0^inf Re[e^{iuk} phi(u - i/2)] / (u^2 + 1/4) du,
        k = log(S/K),               phi = exp(C(u) + D(u) v),

    but the integral is replaced by a fixed Gauss-Legendre rule on [0, U], shared by all
    states. C(u), D(u) depend only on u and tau, so they are computed once; each state
    (S_i, v_i) enters only through k_i and v_i, so all integrands form one (B, N) matrix
    E_ij = exp(i u_j k_i + C(u_j) + D(u_j) v_i), and each integral is a matrix-vector product.

    Greeks are obtained by differentiating under the integral sign:
        dI/dk = int Re[iu e^{iuk} phi] / (u^2 + 1/4) du,
        dI/dv = int Re[D(u - i/2) e^{iuk} phi] / (u^2 + 1/4) du,
        delta = dC/dS = 1 - (1/pi) sqrt(K/S) (I/2 + dI/dk),
        vega  = dC/dv = -sqrt(SK)/pi * dI/dv      (sensitivity to the variance v, not to sqrt(v)).

    S, v:    tensors of shape (B,): current spot and instantaneous variance of each path
             (v is clamped at 0, since the simulator can produce slightly negative values)
    tau:     float, time to maturity T - t_k in years (> 0); the same for all paths on a date
    K:       strike
    params:  HestonParams from market.simulators (alpha, b, sigma, rho are used;
             s0 and v0 are ignored, the current state is given by S and v). Defaults to HestonParams().
    N:       number of quadrature nodes

    The truncation point U grows like 1/sqrt(tau): close to maturity the integrand decays
    slowly, and a fixed U = 200 would give errors of order sqrt(SK)/(pi U) ~ 0.16.

    Returns: (price, delta, vega), each a tensor of shape (B,) with the dtype of S.
    """
    p = HestonParams() if params is None else params
    kappa, theta, xi, rho = p.alpha, p.b, p.sigma, p.rho      # simulator names -> formula names

    ## torch -> numpy (float64), clamp negative variances
    S_np = S.detach().double().numpy()
    v_np = np.clip(v.detach().double().numpy(), 0.0, None)

    ## points and weights on [-1, 1], mapped to [0, U]; U grows as tau shrinks
    U = min(60.0 / math.sqrt(tau * theta), 5000.0)
    x, w = np.polynomial.legendre.leggauss(N)
    u = U * (x + 1) / 2
    wu = w * U / 2 ## new weights for u

    ## first deal with C(u), D(u) since they are indep from the state
    z = u - 0.5j
    beta = kappa - rho * xi * 1j * z
    d = np.sqrt(beta**2 + xi**2 * (1j * z + z**2))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * tau)
    Cu = kappa * theta / xi**2 * ((beta - d) * tau - 2 * np.log((1 - g * e) / (1 - g)))
    Du = (beta - d) / xi**2 * (1 - e) / (1 - g * e)

    ## The integrand at all states, E_ij = exp(i u_j k_i + C(u_j) + D(u_j) v_i), shape (B, N)
    k = np.log(S_np / K)[:, None] #(B, 1)
    E = np.exp(1j * u[None, :] * k + Cu[None, :] + Du[None, :] * v_np[:, None])
    den = u ** 2 + 0.25

    ## The B integration can be solved by matrix multiplication
    I = (np.real(E) / den) @ wu
    ## assuming smoothness so interchange differentiation and integration
    Ik = (np.real(1j * u[None, :] * E) / den) @ wu
    Iv = (np.real(Du[None, :] * E) / den) @ wu

    ## Calculate the greeks and prices
    price = S_np - np.sqrt(S_np * K) / np.pi * I
    delta = 1 - np.sqrt(K / S_np) / np.pi * (0.5 * I + Ik)
    vega = - np.sqrt(S_np * K) / np.pi * Iv

    to_tensor = lambda a: torch.as_tensor(a, dtype=S.dtype, device=S.device)
    return to_tensor(price), to_tensor(delta), to_tensor(vega)
