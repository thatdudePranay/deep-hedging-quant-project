import torch
from benchmarks import hedge_pnl, constant_policy
from market.simulators import simulate_gbm

def test_constant_policy_matches_closed_form():
    torch.manual_seed(0)
    S = simulate_gbm(1000)["S"][:, :, 0]
    T = 30 / 365
    pnl = hedge_pnl(S, constant_policy(0.5), K=100.0, T=T, c=0.0)
    expected = -(S[:, -1] - 100).clamp(min=0) + 0.5 * (S[:, -1] - S[:, 0])
    assert torch.allclose(pnl, expected, atol=1e-4)

def test_two_step_example():
    def policy():
        def two_step_policy(S_k, tau_k):
            return torch.full_like(S_k, 0.5 if tau_k > 1.5 else 0.6)
        return two_step_policy
    
    S_test = torch.tensor([[100., 102., 99.]])
    pnl = hedge_pnl(S_test, policy(), K = 100, T = 2, c = 0.001)
    expected = torch.tensor([-0.9196])
    assert torch.allclose(pnl, expected, atol = 1e-5)
    