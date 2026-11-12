"""
In this objectives.py file we specify the different things our model is trying to optimize
Ergo, its objectives

terminal_pnl   : turns (prices, positions, payoff, costs) into one P&L number per path
CVaRLoss       : trainable CVaR (needs gradients, used in train.py)
EntropicLoss   : exponential-utility risk measure
empirical_cvar : plain CVaR for reporting results (no gradients)

"""

import torch 
import torch.nn as nn

def terminal_PnL(s, deltas, payoff, eps):
    # this is the trading gains: position * price change, summed over dates and instruments
    price_change = s[:, 1:] - s[:, :-1]                       # (B, n, d)
    gains = (deltas * price_change).sum(dim=(1, 2))           # (B,)
 
    # trading costs: size of each trade = change in position
    zeros = torch.zeros_like(deltas[:, :1])                   # delta_{-1} = 0
    prev = torch.cat([zeros, deltas[:, :-1]], dim=1)          # (B, n, d) yesterday's position
    trade = deltas - prev
    eps = torch.as_tensor(eps, dtype=s.dtype, device=s.device)
    cost = (eps * trade.abs() * s[:, :-1]).sum(dim=(1, 2))    # (B,)
 
    return -payoff + gains - cost



##this is for the expected shortfall loss in a trainable form from the paper
##3.9 example

##rho(x) = min_w w + E[ max(-X-w, 0) ] / (1 - alpha)

##w is a added number it learns and at the optimum it equals the VaR.
##    alpha near 0 -> risk neutral, alpha near 1 -> focus on worst cases.
##    The minimised value is the risk-adjusted price of the liability.

class CVaRLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, pnl):
        return self.w + torch.relu(-pnl - self.w).mean() / (1 - self.alpha)

##rho(X) = (1/lambda) * log E[exp(-lambda * X)]  (exponential utility).
class EntropicLoss(nn.Module):
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam
 
    def forward(self, pnl):
        n = torch.tensor(float(pnl.numel()), device=pnl.device)
        return (torch.logsumexp(-self.lam * pnl, dim=0) - torch.log(n)) / self.lam

##Average loss in the worst (1 - alpha) fraction of paths. For reporting only.
@torch.no_grad() ## this just acts as a decorator that turns off gradient calc for below
def empirical_cvar(pnl, alpha: float):
    k = max(1, int((1 - alpha) * pnl.numel()))
    return (-pnl).topk(k).values.mean()  
