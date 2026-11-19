import torch
from market.pricing import bs_delta

def make_features(S_k, tau_k, delta_prev, K, T, s_scale=1.0, V_k=None, v_scale=0.04):
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
    if V_k is not None:
        features.append(V_k / v_scale)
    return torch.stack(features, dim = -1)

def nn_policy(model, K, T, use_prev=False, s_scale=1.0, use_var=False, v_scale=0.04):
    """Wrap a network as a policy(S_k, tau_k, delta_prev, V_k) -> delta_k.

    use_prev: feed delta_prev as a feature (needed with transaction costs;
              model must then have d_in = 3).
    s_scale:  normalization for log(S/K); must be the same value at
              training and evaluation time.
    """
    def policy(S_k, tau_k, delta_prev, V_k = None):
        dp = delta_prev if use_prev else None
        var = V_k if use_var else None
        features = make_features(S_k, tau_k, dp, K, T, s_scale, var, v_scale)
        delta = model(features)
        return delta.squeeze(-1)
    return policy


def ntb_policy(model, K, T, sigma, s_scale = 1.0):
    """No-transaction-band policy: policy(S_k, tau_k, delta_prev, V_k) -> delta_k.

    The network does not output the position. It outputs the band around the
    BS delta, and the position is obtained by clamping the previous holding:

        lo = Delta_BS(S, tau) - softplus(o1),   hi = Delta_BS(S, tau) + softplus(o2),
        delta_k = clamp(delta_prev, lo, hi).

    Inside the band nothing is traded; outside, trade only back to the nearest edge
    (same rule as ww_band_policy, but the two widths are learned and may differ).
    delta_prev is not a network input, it is only used in the clamp.

    model:    network with d_in = 2 (log-moneyness, time) and m_out = 2 (o1, o2).
              Initialize the last bias to about -3 so the initial band is narrow;
              a wide initial band contains delta_prev = 0, no trade happens and the
              gradient is zero.
    sigma:    volatility used for the BS delta at the centre of the band
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