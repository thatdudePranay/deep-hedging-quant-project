import torch
from market import simulators

S = simulators.simulate_gbm(100_000)["S"]
print(S.shape)                                        # (100000, 31, 1)
print(S[:, -1, 0].mean())                             # about 100
print(torch.log(S[:, -1, 0] / 100).std())             # about 0.0573

H = simulators.simulate_heston(50_000)
print(H["S"][:, -1, 0].mean())                        # about 100
print(H["V"].mean(), (H["V"] <= 0).float().mean())    # variance average, and share of paths at or below zero
