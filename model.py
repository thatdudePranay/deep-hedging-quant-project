import torch
from torch import nn


"""MLP hedging policy. forward: (B, d_in) -> (B, m_out), no output activation."""
class Policy(nn.Module):
    def __init__(self, d_in, m_out, hidden = (64, 64), activiation = nn.ReLU):
        ## hidden: depth and width of the NN
        super().__init__()
        layers = []
        prev = d_in ## the output dimension of prev layer

        ## building NN
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(activiation())
            prev = h
        layers.append(nn.Linear(prev, m_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
