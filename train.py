import torch
from features import nn_policy
from benchmarks import hedge_pnl
from market.simulators import simulate_gbm

def train(model, K, T, loss_fn=None, policy=None, n_steps=2000, B=5000, lr=1e-3, c=0.0,
          use_prev=False, s_scale=1.0, sim_kwargs=None, print_every=100):
    """Train a hedging network by simulating fresh GBM paths at every step.

    Two modes, depending on loss_fn:
      * loss_fn=None (quadratic): minimize E[(PnL + p0)^2] jointly over the network
        and a learnable premium p0 (p0 has its own learning rate 1e-2).
        At the optimum p0 = -E[PnL], i.e. the quadratic-hedging price.
      * loss_fn = EntropicLoss / CVaRLoss (cash-invariant risk measures): minimize
        rho(PnL) with the premium excluded; no p0 is learned (it would diverge,
        since rho(X + p0) = rho(X) - p0). The price is rho(PnL) itself, to be
        evaluated afterwards on a separate test set. Parameters inside loss_fn
        (e.g. w of CVaRLoss) are optimized together with the network.

    model:      the network (e.g. model.Policy); updated in place
    K, T:       strike and maturity (years); T must equal n_steps * dt of the simulator
    loss_fn:    None (quadratic) or a risk-measure module from objectives.py
    policy:     optional policy(S_k, tau_k, delta_prev, V_k) built from `model`
                (e.g. features.ntb_policy). If None, nn_policy(model, K, T, use_prev, s_scale)
    n_steps:    number of gradient steps; B: paths per step; lr: learning rate of the network
    c:          proportional transaction cost rate
    use_prev:   feed delta_prev as a feature (only used when policy is None)
    s_scale:    normalization of log(S/K) (only used when policy is None)
    sim_kwargs: extra keyword arguments for simulate_gbm (e.g. {"sigma": 0.3})
    print_every: print the loss every this many steps

    Returns a dict:
        "model":   the trained network (same object as the input)
        "p0":      learned premium (nn.Parameter); meaningful only in quadratic mode
        "loss_fn": the risk-measure module (None in quadratic mode)
        "losses":  list of training losses, one per step
    """
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