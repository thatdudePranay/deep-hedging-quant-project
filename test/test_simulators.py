import torch
from market.simulators import simulate_gbm
from market.pricing import bs_price, bs_delta, bs_gamma

def test_gbm_call_price():
    torch.manual_seed(0)
    S = simulate_gbm(100000)["S"]
    B = S.shape[0]
    K, tau, sigma = 100, 30/365, 0.2
    sim_payoff = (S[:, -1, 0] - K).clamp(min = 0)
    sim_C = sim_payoff.mean()
    bs_C = bs_price(100, K, tau, sigma)
    se = sim_payoff.std() / (B ** 0.5)
    assert (sim_C - bs_C).abs() < 3 * se

def test_bs_delta_finite_difference():
    h = 0.01
    p1 = bs_price(100 + h, 100, 30/365, 0.2)
    p2 = bs_price(100 - h, 100, 30/365, 0.2)
    delta = bs_delta(100, 100, 30/365, 0.2)
    est_delta = (p1 - p2) / (2 * h)
    assert (est_delta - delta).abs() < 1e-04

def test_bs_gamma_finite_difference():
    h = 0.01
    d1 = bs_delta(100 + h, 100, 30/365, 0.2)
    d2 = bs_delta(100 - h, 100, 30/365, 0.2)
    gamma = bs_gamma(100, 100, 30/365, 0.2)
    est_gamma = (d1 - d2) / (2 * h)
    assert (est_gamma - gamma).abs() < 1e-04
