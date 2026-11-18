import torch
from features import nn_policy
from benchmarks import hedge_pnl
from market.simulators import simulate_gbm

def train(model, K, T, loss_fn=None, policy=None, n_steps=2000, B=5000, lr=1e-3, c=0.0,
          use_prev=False, s_scale=1.0, sim_kwargs=None, print_every=100):
    """Train policy + premium p0 under speicifc loss function,
    when not specified, use quadratic loss. 
    Returns (model, p0, losses)."""
    if sim_kwargs is None:
        sim_kwargs = {}
    p0 = torch.nn.Parameter(torch.zeros(()))

    if policy is None:
        policy = nn_policy(model, K, T, use_prev, s_scale)

    if loss_fn is None:
        opt = torch.optim.Adam([
            {"params": model.parameters(), "lr": lr},
            {"params": [p0], "lr": 1e-2}
            ])
    else: ## do not learn p0 when not quadratic loss
        opt = torch.optim.Adam([
            {"params": model.parameters(), "lr": lr},
            {"params": loss_fn.parameters(), "lr": 1e-2}
            ])

    losses = []
    for step in range(n_steps):
        S = simulate_gbm(B, **sim_kwargs)["S"][:, :, 0]

        if loss_fn is None:
            pnl = hedge_pnl(S, policy, K, T, c=c, p0=p0)
            loss = (pnl ** 2).mean() ## default quadratic loss
        else:
            pnl = hedge_pnl(S, policy, K, T, c=c, p0=0.0)
            loss = loss_fn(pnl)

        ## Reporting
        losses.append(loss.item())
        if step % print_every == 0:
            print(f"step: {step}, loss: {loss.item()}")

        ## Training and updating
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    return {"model": model, "loss_fn": loss_fn, "losses": losses, 'p0': p0}