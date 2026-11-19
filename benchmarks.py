"""Benchmark hedging strategies and evaluation tools for the deep hedging comparison.

Every policy is built by a factory function and has the common signature

    policy(S_k, tau_k, delta_prev, V_k) -> delta_k

    S_k         (B,)   stock price at date k
    tau_k       float  time to maturity T - t_k, in years
    delta_prev  (B,)   holding chosen at the previous date (0 at k = 0)
    V_k         (B,)   instantaneous variance at date k (only used by stochastic-vol policies)
    delta_k     (B,)   holding chosen at date k

With d > 1 hedging instruments (Heston: stock + variance swap), delta_prev and delta_k
have shape (B, d) instead of (B,); S_k is always the stock price.

Policies:   constant_policy, bs_delta_policy, ww_band_policy, bs_delta_stovol_policy,
            heston_complete_policy (two instruments)
Evaluation: hedge_pnl (rollout), summarize (risk metrics)
All formulas assume r = 0.
"""

import torch, math
from market.pricing import bs_delta, bs_gamma, heston_greeks, HestonParams


# ---------------------------------------------------------------------------
# Policies, BS
# ---------------------------------------------------------------------------

def constant_policy(a):
    """Hold a fixed number `a` of shares on every path at every date (used for testing)."""
    def policy(S_k, tau_k, delta_prev, V_k=None):
        return torch.full_like(S_k, a)
    return policy


def bs_delta_policy(K, sigma):
    """Black-Scholes delta hedge with a known, constant volatility sigma."""
    def policy(S_k, tau_k, delta_prev, V_k=None):
        return bs_delta(S_k, K, tau_k, sigma)
    return policy


def ww_band_policy(K, sigma, c, gamma):
    """Whalley-Wilmott (1997) no-trade band around the BS delta, for proportional costs c.

    Idea: only trade when the current holding is too far from the BS delta, and then
    only trade back to the edge of the band. The band is [Delta_k - H_k, Delta_k + H_k] with

        H_k = (3 c S_k Gamma_k^2 / (2 gamma))^(1/3),

    where gamma is the risk-aversion coefficient: larger gamma -> narrower band -> more trading.
    With c = 0 the band has zero width and the policy reduces to the BS delta hedge.
    """
    def policy(S_k, tau_k, delta_prev, V_k=None):
        Gamma_k = bs_gamma(S_k, K, tau_k, sigma)
        H_k = ((3 * c * S_k * Gamma_k ** 2) / (2 * gamma)) ** (1 / 3)
        bs_delta_k = bs_delta(S_k, K, tau_k, sigma)
        return torch.clamp(delta_prev, min=bs_delta_k - H_k, max=bs_delta_k + H_k)
    return policy


def bs_delta_stovol_policy(K, sigma_floor=1e-4):
    """Black-Scholes delta hedge using the current instantaneous volatility."""
    def policy(S_k, tau_k, delta_prev, V_k):
        sigma_k = torch.sqrt(torch.clamp(V_k, min = 0)).clamp(min = sigma_floor)
        return bs_delta(S_k, K, tau_k, sigma_k)
    return policy


# ---------------------------------------------------------------------------
# Policies, Heston
# ---------------------------------------------------------------------------
def heston_complete_policy(K, params=None):
    """Heston complete-market hedge with two instruments: stock and variance swap (c = 0).

    The call price C(S, v, tau) has two risk factors, S and v. Hedge each with one instrument:
        stock:         delta_1 = dC/dS                            (Heston delta)
        variance swap: delta_2 = (dC/dv) / (dL/dv),   dL/dv = (1 - exp(-alpha tau)) / alpha,
    where L(t, v) is the remaining-variance term of the variance swap (simulators.simulate_heston),
    so delta_2 cancels the sensitivity of the call to the instantaneous variance.
    Exact in continuous time; with discrete rebalancing a small hedging error remains.

    params: HestonParams (defaults to HestonParams()); must match the simulated paths.
    Returns policy(S_k, tau_k, delta_prev, V_k) -> (B, 2). delta_prev is not used.
    Use with hedge_pnl on S of shape (B, n+1, 2) and V passed explicitly.
    """
    p = HestonParams() if params is None else params
    def policy(S_k, tau_k, delta_prev, V_k):
        _, delta, vega = heston_greeks(S_k, V_k, tau_k, K, p)
        alpha = p.alpha
        dLdv = (1 - math.exp(-alpha * tau_k)) / alpha
        return torch.stack([delta, vega / dLdv], dim = -1)
    return policy

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def hedge_pnl(S, policy, K, T, c=0.0, p0=0.0, V=None):
    """Terminal P&L of a short European call hedged with `policy` (r = 0).

        PnL = p0 - (S_T - K)^+ + sum_k delta_k (S_{k+1} - S_k) - transaction costs

    The P&L covers the whole short-call position (liability included), not just the hedge.
    The premium p0 is 0 by default, i.e. the P&L excludes the premium unless one is passed.

    S:          shape (B, n+1), B simulated paths of the stock, or
                shape (B, n+1, d) for d hedging instruments; instrument 0 must be the stock
                (it determines the call payoff and is passed to the policy as S_k)
    policy:     hedging strategy, called at each date k = 0, ..., n-1 as
                    policy(S_k, tau_k, delta_prev, V_k) -> delta_k
                with S_k the stock price (B,). For d = 1, delta_prev and delta_k have shape (B,)
                (all single-instrument policies); for d > 1 they have shape (B, d).
                (see the module docstring for the argument shapes)
    K, T, c:    strike, maturity (years), proportional cost rate (same rate for every
                instrument: cost = c * sum_j S_j |delta_j - delta_prev_j|)
    p0:         premium received at t0 (e.g. the BS price under GBM, the Heston price under Heston)
    V:          optional, shape (B, n+1): instantaneous variance paths, passed to the policy as
                extra information only (it does not enter the P&L accounting). Defaults to zeros,
                which is fine for policies that ignore V (constant, BS delta, WW band) but wrong
                for policies that use it (e.g. stochastic-vol BS delta): pass it explicitly as V=out["V"].

    Returns:    shape (B,)
    """
    if S.dim() == 2:
        S = S.unsqueeze(-1)

    _, n_plus_1, d = S.shape

    n = n_plus_1 - 1
    dt = T / n
    if V is None:
        V = torch.zeros_like(S[:, :, 0])

    pnl = torch.zeros_like(S[:, 0, 0])
    delta_prev = torch.zeros_like(S[:, 0, :])          # start flat

    # trade at each date k = 0, ..., n-1: gain over [t_k, t_{k+1}] minus cost of rebalancing
    for k in range(n):
        tau_k = T - k * dt
        if d == 1:
            dp_in = delta_prev.squeeze(-1)          
            delta = policy(S[:, k, 0], tau_k, dp_in, V[:, k]).unsqueeze(-1)
        else:
            delta = policy(S[:, k, 0], tau_k, delta_prev, V[:, k])

        pnl = pnl + (delta * (S[:, k+1, :] - S[:, k, :])).sum(-1) \
            - c * (S[:, k, :] * (delta - delta_prev).abs()).sum(-1)
        delta_prev = delta

    # at maturity: liquidate the hedge (cost) and pay the call payoff
    pnl = pnl - (c * S[:, -1, :] * delta_prev.abs()).sum(-1) - (S[:, -1, 0] - K).clamp(min=0)
    return pnl + p0


def summarize(pnl, alpha=0.95):
    """Summary statistics of a P&L sample (input is P&L, not loss).

    mean, std:  of the P&L
    VaR, CVaR:  at level alpha, computed on the loss L = -P&L
                (VaR = alpha-quantile of L, CVaR = mean of L over L >= VaR)
    """
    L = -pnl
    VaR = torch.quantile(L, alpha).item()
    return {'mean': pnl.mean().item(), 'std': pnl.std().item(),
            'VaR': VaR, 'CVaR': L[L >= VaR].mean().item()}
