import torch
from market.simulators import simulate_gbm
from market.pricing import bs_price

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