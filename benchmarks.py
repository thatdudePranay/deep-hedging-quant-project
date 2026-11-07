""" Benchmark hedging strategies for comparison study with deep hedging

Two strategies used as benchmark: constant policy, bs_delta policy
"""

import torch
from market.pricing import bs_delta

def constant_policy(a):
    """Constant longing/shorting for one instrument for all paths on day k"""
    def policy(S_k, tau_k):
        """
        S_k:    vector of length B, the price of instrument for all paths on day k
        tau_k:  time to maturity, measured in years
        """
        return torch.full_like(S_k, a)
    return policy


def bs_delta_policy(K, sigma):
    """BS delta hedge, assuming 0 interest rate and known volatility"""
    def policy(S_k, tau_k):
        return bs_delta(S_k, K, tau_k, sigma)
    return policy


def hedge_pnl(S, policy, K, T, c = 0.0, p0 = 0.0):
    """Terminal P&L of a short European call hedged with `policy` (r = 0).

        PnL = p0 - (S_T - K)^+ + sum_k delta_k (S_{k+1} - S_k) - transaction costs

    The P&L covers the whole short-call position (liability included), not just the hedge.
    The premium p0 is 0 by default, i.e. the P&L excludes the premium unless one is passed.

    S:          shape (B, n+1), B simulated paths of the stock
    policy:     hedging strategy, policy(S_k, tau_k) -> delta_k of shape (B,)
    K, T, c:    strike, maturity (years), proportional cost rate
    p0:         premium received at t0 (e.g. the BS price)

    Returns:    shape (B,)
    """
    _, n_plus_1 = S.shape
    n = n_plus_1 - 1
    dt = T / n
    pnl, delta_prev = torch.full_like(S[:,0], 0), torch.full_like(S[:,0], 0)

    ## Calculate the pnl on each trading date
    for k in range(n):
        S_k = S[:, k]
        tau_k = T - k * dt
        delta = policy(S_k, tau_k)
        pnl = pnl + delta * (S[:, k+1] - S_k) - c * (delta - delta_prev).abs() * S_k
        delta_prev = delta
    
    ## Add the final day value (sell all holdings, count the payoff of call)
    pnl = pnl - c * S[:, -1] * (delta_prev).abs() - (S[:, -1] - K).clamp(min = 0)
    return pnl + p0

def summarize(pnl, alpha = 0.95):
    """Calculate VaR and cVaR at alpha
    
    Note that the input is pnl not loss"""
    L = -pnl
    VaR = torch.quantile(L, alpha).item()
    return {'mean': pnl.mean().item(), 'std': pnl.std().item(),
            'VaR': VaR, 'CVaR': L[L >= VaR].mean().item()}
