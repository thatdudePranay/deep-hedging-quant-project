import torch
import benchmarks as bm
from market.simulators import simulate_gbm

def test_constant_policy_matches_closed_form():
    torch.manual_seed(0)
    S = simulate_gbm(1000)["S"][:, :, 0]
    T = 30 / 365
    pnl = bm.hedge_pnl(S, bm.constant_policy(0.5), K=100.0, T=T, c=0.0)
    expected = -(S[:, -1] - 100).clamp(min=0) + 0.5 * (S[:, -1] - S[:, 0])
    assert torch.allclose(pnl, expected, atol=1e-4)

def test_two_step_example():
    def policy():
        def two_step_policy(S_k, tau_k, delta_prev, V = None):
            return torch.full_like(S_k, 0.5 if tau_k > 1.5 else 0.6)
        return two_step_policy
    
    S_test = torch.tensor([[100., 102., 99.]])
    pnl = bm.hedge_pnl(S_test, policy(), K = 100, T = 2, c = 0.001)
    expected = torch.tensor([-0.9196])
    assert torch.allclose(pnl, expected, atol = 1e-5)

def test_naive_summary():
    pnl = torch.tensor([-2, -3, 0, -6, 1], dtype = float)
    out = bm.summarize(pnl, 0.6)
    assert abs(out['CVaR'] - 4.5) < 1e-6 and  2 <= out['VaR'] <= 3

def test_ww_band_no_cost():
    torch.manual_seed(0)
    S = simulate_gbm(1000)["S"][:, :, 0]
    T = 30/365
    bs_pnl = bm.hedge_pnl(S, bm.bs_delta_policy(100, 0.2), 100, T)
    ww_pnl = bm.hedge_pnl(S, bm.ww_band_policy(100, 0.2, 0, 1), 100, T)
    assert torch.allclose(bs_pnl, ww_pnl, atol = 1e-4)

def test_bs_stovol_on_gbm():
    torch.manual_seed(0)
    out = simulate_gbm(1000)
    S, V = out["S"][:, :, 0], out["V"]

    T = 30/365
    bs_pnl = bm.hedge_pnl(S, bm.bs_delta_policy(100, 0.2), 100, T)
    bs_stovol_pnl = bm.hedge_pnl(S, bm.bs_delta_stovol_policy(100), 100, T, V = V)
    assert torch.allclose(bs_pnl, bs_stovol_pnl, atol = 1e-4)