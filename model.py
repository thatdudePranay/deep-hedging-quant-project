import torch
from torch import nn


class Policy(nn.Module):
    """Fully connected network (MLP) used as the hedging policy.

    forward: (B, d_in) -> (B, m_out), linear output (no final activation).
    The same network is shared across all dates k; time enters through the features.

    d_in:        number of input features (2 without costs, 3 with delta_prev)
    m_out:       number of outputs (1 = position in the stock; 2 = two hedging
                 instruments, or the two band widths of the no-transaction-band policy)
    hidden:      widths of the hidden layers, e.g. (64, 64) = two layers of 64 units
    activiation: activation class applied after every hidden layer (default nn.ReLU)

    Example: Policy(3, 1) has 4481 trainable parameters.
    """
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
        """x: (B, d_in) features -> (B, m_out) raw network output."""
        return self.net(x)
