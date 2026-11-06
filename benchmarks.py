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


def hedge_pnl(S, policy, K, T, c = 0.0):
    """Calculate the hedging pnl
    
    S:          shape (B, n+1), B simulated paths of stocks
    policy:     hedging strategy
    K, T, c:    strike, maturity, cost rate
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
    return pnl
