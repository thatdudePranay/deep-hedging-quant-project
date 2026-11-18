import torch

def make_features(S_k, tau_k, K, T):
    remain_t = torch.full_like(S_k, tau_k / T)
    return torch.stack([torch.log(S_k / K), remain_t], dim = -1)

def nn_policy(model, K, T):
    def policy(S_k, tau_k, delta_prev, V_k = None):
        features = make_features(S_k, tau_k, K, T)
        delta = model(features)
        return delta.squeeze(-1)
    return policy