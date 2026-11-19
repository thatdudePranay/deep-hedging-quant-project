import torch
from features import nn_policy
from benchmarks import hedge_pnl
from market.simulators import simulate_gbm

def train(model, K, T, loss_fn=None, policy=None, n_steps=2000, B=5000, lr=1e-3, c=0.0,
          use_prev=False, s_scale=1.0, sim_kwargs=None, print_every=100,
          use_var=False, v_scale=0.04, simulator=simulate_gbm, n_instr=1, out_scale=1.0):
    """Train a hedging network by simulating fresh market paths at every step.

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
                (e.g. features.ntb_policy). If None, it is built as
                nn_policy(model, K, T, use_prev, s_scale, use_var, v_scale, out_scale)
    n_steps:    number of gradient steps; B: paths per step; lr: learning rate of the network
    c:          proportional transaction cost rate (applied to every instrument)
    use_prev:   feed delta_prev as a feature (only used when policy is None)
    s_scale:    normalization of log(S/K) (only used when policy is None)
    sim_kwargs: extra keyword arguments for the simulator
                (e.g. {"sigma": 0.3} for GBM, {"params": HestonParams()} for Heston)
    print_every: print the loss every this many steps
    use_var:    feed V_k / v_scale as a feature (only used when policy is None)
    v_scale:    normalization of V_k (only used when policy is None)
    simulator:  function(n_paths, **sim_kwargs) -> dict with "S" (B, n+1, d) and "V" (B, n+1),
                e.g. simulate_gbm (default) or simulate_heston
    n_instr:    number of hedging instruments: 1 = stock only (S[:, :, 0]),
                otherwise all instruments of the simulator (Heston: stock + variance swap)
    out_scale:  scale of the network output (only used when policy is None), see nn_policy

    Note: s_scale, v_scale, out_scale, use_prev, use_var must be passed with the same values
    to nn_policy when the trained model is evaluated.

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
        policy = nn_policy(model, K, T, use_prev, s_scale, use_var, v_scale, out_scale)

    if loss_fn is None: ## default quadratic loss
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
        out = simulator(B, **sim_kwargs)
        V = out['V']
        S = out['S'][:, :, 0] if n_instr == 1 else out['S']

        if loss_fn is None:
            pnl = hedge_pnl(S, policy, K, T, c=c, p0=p0, V=V)
            loss = (pnl ** 2).mean() ## default quadratic loss
        else:
            pnl = hedge_pnl(S, policy, K, T, c=c, p0=0.0, V=V)
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