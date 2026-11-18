import torch
from features import nn_policy
from benchmarks import hedge_pnl
from market.simulators import simulate_gbm

def train(model, K, T, n_steps=2000, B=5000, lr=1e-3, c=0.0,
          sim_kwargs=None, print_every=100):
    """Train policy + premium p0 under quadratic loss.
    Returns (model, p0, losses)."""
    if sim_kwargs is None:
        sim_kwargs = {}
    p0 = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": lr},
        {"params": [p0], "lr": 1e-2}
        ])
    policy = nn_policy(model, K, T)

    losses = []
    for step in range(n_steps):
        S = simulate_gbm(B, **sim_kwargs)["S"][:, :, 0]
        pnl = hedge_pnl(S, policy, K, T, c, p0)
        loss = (pnl ** 2).mean()

        ## Reporting
        losses.append(loss.item())
        if step % print_every == 0:
            print(f"step: {step}, loss: {loss.item()}, p0: {p0.item()}")

        ## Training and updating
        opt.zero_grad()
        loss.backward()
        opt.step()

    return {"model": model, "p0": p0, "losses": losses}