import torch
from market.simulators import HestonParams
from market.pricing import heston_price, heston_greeks


def test_heston_greeks_price_matches_quad():
    T = 30 / 365
    S = torch.tensor([100.0], dtype=torch.float64)
    v = torch.tensor([HestonParams().v0], dtype=torch.float64)
    price, _, _ = heston_greeks(S, v, T, K=100.0)
    assert abs(price.item() - heston_price(K=100.0, T=T)) < 1e-3


def test_heston_greeks_finite_difference():
    tau, K = 10 / 365, 100.0
    S = torch.tensor([95.0, 100.0, 105.0], dtype=torch.float64)
    v = torch.tensor([0.01, 0.04, 0.1], dtype=torch.float64)
    h, hv = 1e-3, 1e-5
    _, dS, dv = heston_greeks(S, v, tau, K)
    fd_S = (heston_greeks(S + h, v, tau, K)[0] - heston_greeks(S - h, v, tau, K)[0]) / (2 * h)
    fd_v = (heston_greeks(S, v + hv, tau, K)[0] - heston_greeks(S, v - hv, tau, K)[0]) / (2 * hv)
    assert torch.allclose(dS, fd_S, atol=1e-5)
    assert torch.allclose(dv, fd_v, atol=1e-4)
