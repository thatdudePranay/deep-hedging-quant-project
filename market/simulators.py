"""
Here we'll have the simulators for the model, that will generate the market & data that will be used to train the model. 
The simulators will be based on the market dynamics and will generate synthetic data that mimics real-world market behavior. 
"""

from dataclasses import dataclass
import math
import torch

@dataclass
class HestonParams:
    # mean-reversion speed
    alpha: float = 1.0
    # long-run variance
    b: float = 0.04
    # spot/variance correlation
    rho: float = -0.7
    # vol of variance
    sigma: float = 2.0
    v0: float = 0.04
    s0: float = 100.0


def simulate_heston(
    n_paths: int,
    n_steps: int = 30,
    dt: float = 1 / 365,
    params: HestonParams = HestonParams(),
    substeps: int = 10,
    device: str = "cpu",
):
    """Returns dict with
        S : (n_paths, n_steps+1, 2)  hedging instruments [spot, variance swap]
        V : (n_paths, n_steps+1)     instantaneous variance (a feature)
    """
    p = params
    T = n_steps * dt
    h = dt / substeps

    # variance-swap "remaining" term, paper eq. (5.4)
    def L(t, v):  
        tau = T - t
        return (v - p.b) / p.alpha * (1 - math.exp(-p.alpha * tau)) + p.b * tau

    s = torch.full((n_paths,), p.s0, device=device)
    v = torch.full((n_paths,), p.v0, device=device)
    iv = torch.zeros(n_paths, device=device) 

    S_out = [torch.stack([s, iv + L(0.0, v.clamp(min=0))], dim=-1)]
    V_out = [v.clone()]

    for k in range(n_steps):
        for _ in range(substeps):
            z1 = torch.randn(n_paths, device=device)
            z2 = torch.randn(n_paths, device=device)
            zs = p.rho * z1 + math.sqrt(1 - p.rho ** 2) * z2
            vp = v.clamp(min=0)
            iv = iv + vp * h
            s = s * torch.exp(-0.5 * vp * h + torch.sqrt(vp * h) * zs)
            v = v + p.alpha * (p.b - vp) * h + p.sigma * torch.sqrt(vp * h) * z1
        t = (k + 1) * dt
        S_out.append(torch.stack([s, iv + L(t, v.clamp(min=0))], dim=-1))
        V_out.append(v.clone())

    return {"S": torch.stack(S_out, dim=1), "V": torch.stack(V_out, dim=1), "T": T, "dt": dt}


def call_payoff(S, strike: float):
    """European call on the spot (instrument 0). we shud replace with any G(S_0..S_T)."""
    return torch.clamp(S[:, -1, 0] - strike, min=0.0)
