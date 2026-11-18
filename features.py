import torch
from market.pricing import bs_delta

def make_features(S_k, tau_k, delta_prev, K, T, s_scale=1.0):
    """Build the network input at date k.

    Features (last dim):
        log(S_k / K) / s_scale   -- moneyness, normalized so it is O(1)
        tau_k / T                -- fraction of time to maturity remaining
        delta_prev               -- previous position (only if not None)

    s_scale: typical size of log-moneyness moves, e.g. sigma * sqrt(T).
             Default 1.0 = no normalization.
    Returns a tensor of shape (B, 2) or (B, 3).
    """
    remain_t = torch.full_like(S_k, tau_k / T)
    features = [torch.log(S_k / K) / s_scale, remain_t]
    if delta_prev is not None:
        features.append(delta_prev)
    return torch.stack(features, dim = -1)

def nn_policy(model, K, T, use_prev = False, s_scale = 1.0):
    """Wrap a network as a policy(S_k, tau_k, delta_prev, V_k) -> delta_k.

    use_prev: feed delta_prev as a feature (needed with transaction costs;
              model must then have d_in = 3).
    s_scale:  normalization for log(S/K); must be the same value at
              training and evaluation time.
    """
    def policy(S_k, tau_k, delta_prev, V_k = None):
        dp = delta_prev if use_prev else None
        features = make_features(S_k, tau_k, dp, K, T, s_scale)
        delta = model(features)
        return delta.squeeze(-1)
    return policy


def ntb_policy(model, K, T, sigma, s_scale = 1.0):
    """Wrap a network as a policy(S_k, tau_k, delta_prev V_k) -> delta_k.
    Different from nn_policy, we do not directly pass delta_prev
    but consider a no-transaction-band like WW-policy.

    model:    output must have d_in = d_out = 2
    s_scale:  normalization for log(S/K); must be the same value at
              training and evaluation time.
    """
    def policy(S_k, tau_k, delta_prev, V_k = None):
        features = make_features(S_k, tau_k, None, K, T, s_scale)
        out = model(features)
        o1, o2 = out[:, 0], out[:, 1]
        bs = bs_delta(S_k, K, tau_k, sigma)
        lo = bs - torch.nn.functional.softplus(o1)
        hi = bs + torch.nn.functional.softplus(o2)
        return torch.clamp(delta_prev, min = lo, max = hi)
    return policy