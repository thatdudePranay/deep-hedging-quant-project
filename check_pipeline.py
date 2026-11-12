## just a test for how the pipeline is working so far:
## no neural network yet
##expecteiing the following:
## - 'no hedge' has a large std (the whole payoff is risk).
## - 'BS delta, eps=0' has a std several times smaller (hedging works).
## - Raising eps makes the hedged result worse (costs bite).


from market.simulators import simulate_gbm, call_payoff
from objectives import terminal_PnL, empirical_cvar
from benchmarks import no_hedge, bs_delta_hedge
from utils import set_seed
 
set_seed(0)
K = 100.0
sim = simulate_gbm(100_000)
Z = call_payoff(sim["S"], K)
 
cases = [
    ("no hedge",             no_hedge(sim),           0.0),
    ("BS delta, eps=0",      bs_delta_hedge(sim, K),  0.0),
    ("BS delta, eps=0.001",  bs_delta_hedge(sim, K),  0.001),
    ("BS delta, eps=0.01",   bs_delta_hedge(sim, K),  0.01),
]
 
print(f"mean payoff E[Z] (about the fair price): {Z.mean():.4f}\n")
print(f"{'strategy':22s} {'mean PnL':>9s} {'std':>8s} {'CVaR95(loss)':>13s}")

for name, deltas, eps in cases:
    pnl = terminal_PnL(sim["S"], deltas, Z, eps)
    print(f"{name:22s} {pnl.mean():9.4f} {pnl.std():8.4f} {empirical_cvar(pnl, 0.95):13.4f}")
