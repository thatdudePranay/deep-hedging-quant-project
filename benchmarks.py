"""
Here we'll define the benchmarks and the traditional hedges the deep model has to beat

Every function takes the simulation dict and returns positions shaped (B, n, d),
the same format the neural network outputs, so they plug into terminal_pnl identically.

"""

import math
import torch


def no_hedge(sim):
    """Hold nothing. The floor."""
    B, n1, d = sim["S"].shape
    return torch.zeros(B, n1 - 1, d, device=sim["S"].device)
 

## Black-Scholes delta of a call, using the current vol sqrt(V_t).
## Trades the spot only (instrument 0); other instruments get a zero position.
## Under GBM this is the textbook hedge. Under Heston it is only a rough
##  stand-in for the true model hedge.
def bs_delta_hedge(sim, strike: float):
    S1 = sim["S"][:, :-1, 0]                           # (B, n) spot at each decision date
    V = sim["V"][:, :-1].clamp(min=1e-6)               # variance, kept positive
    n = S1.shape[1]
    t = torch.arange(n, device=S1.device) * sim["dt"]
    tau = (sim["T"] - t).unsqueeze(0)                  # (1, n) time left to expiry
 
    d1 = (torch.log(S1 / strike) + 0.5 * V * tau) / (V.sqrt() * tau.sqrt())
    delta1 = 0.5 * (1 + torch.erf(d1 / math.sqrt(2)))  # standard normal CDF of d1
 
    deltas = torch.zeros(S1.shape[0], n, sim["S"].shape[-1], device=S1.device)
    deltas[..., 0] = delta1
    return deltas
