import torch
from market.pricing import bs_delta

def make_features(S_k, tau_k, delta_prev, K, T, s_scale=1.0, V_k=None, v_scale=0.04):
    """Build the network input at date k.

    Features (last dim, in this order):
        log(S_k / K) / s_scale   -- moneyness, normalized so it is O(1)
        tau_k / T                -- fraction of time to maturity remaining
        delta_prev               -- previous position (only if delta_prev is not None)
        V_k / v_scale            -- instantaneous variance, normalized (only if V_k is not None)

    S_k:        (B,) stock price at date k
    tau_k:      float, time to maturity T - t_k (years)
    delta_prev: (B,) previous holding, or None to leave it out
    K, T:       strike and maturity
    s_scale:    typical size of log-moneyness moves, e.g. sigma * sqrt(T) under GBM,
                sqrt(b * T) under Heston. Default 1.0 = no normalization.
    V_k:        (B,) instantaneous variance at date k, or None to leave it out
    v_scale:    typical size of the variance, e.g. the Heston long-run variance b

    Returns a tensor of shape (B, 2), (B, 3) or (B, 4).
    """
    remain_t = torch.full_like(S_k, tau_k / T)
    features = [torch.log(S_k / K) / s_scale, remain_t]
    if delta_prev is not None:
        features.append(delta_prev)
    if V_k is not None:
        features.append(V_k / v_scale)
    return torch.stack(features, dim = -1)

def nn_policy(model, K, T, use_prev=False, s_scale=1.0, use_var=False, v_scale=0.04, out_scale=1.0):
    """Wrap a network as a policy(S_k, tau_k, delta_prev, V_k) -> delta_k.

    The network maps the features of make_features to the position(s):
        delta_k = model(features) * out_scale.

    model:     network with d_in = 2 + use_prev + use_var and m_out = number of instruments
    K, T:      strike and maturity
    use_prev:  feed delta_prev as a feature (needed with transaction costs)
    s_scale:   normalization for log(S/K)
    use_var:   feed V_k / v_scale as a feature (stochastic volatility, e.g. Heston)
    v_scale:   normalization for V_k, e.g. the Heston long-run variance b
    out_scale: scale applied to the network output, so the network can work with O(1)
               numbers while the positions are large; a float or a tensor of shape (m_out,),
               e.g. torch.tensor([1.0, 100.0]) for stock + variance swap
               (the variance-swap position is of order 100s)

    s_scale, v_scale, out_scale (and use_prev, use_var) must be the same at training and
    evaluation time; the easiest way is to keep them in one dict and pass it to both
    train(...) and nn_policy(...).

    Returns policy(S_k, tau_k, delta_prev, V_k) -> delta_k of shape (B,) if m_out = 1,
    else (B, m_out).
    """
    def policy(S_k, tau_k, delta_prev, V_k = None):
        dp = delta_prev if use_prev else None
        var = V_k if use_var else None
        features = make_features(S_k, tau_k, dp, K, T, s_scale, var, v_scale)
        delta = model(features) * out_scale
        return delta.squeeze(-1)
    return policy


# TODO(Micky): fully understand the NTB policy -- why the band structure helps,
#              the softplus widths, the bias initialization, and where the gradient flows.
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