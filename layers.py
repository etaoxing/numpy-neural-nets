import numpy as np

from functions import get_activation

class Linear:
    def __init__(self, in_channels, out_channels, act=None):
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = np.random.randn(out_channels, in_channels) / np.sqrt(in_channels) # NOTE: normal distribution
        self.bias = np.zeros((out_channels, 1)) # NOTE: zeroed out
        self.act = get_activation(act)

    def forward(self, x):
        z = (self.weight @ x) + self.bias
        return z, self.act(z)

    def backward(self, x, z, da):
        dz = da * self.act(z, deriv=True)
        dx = self.weight.T @ dz

        dw = dz @ x.T
        db = np.mean(dz, axis=1, keepdims=True) # NOTE: could also np.sum so not scaling by constant
        return dx, dw, db

    def __str__(self):
        return 'Linear({}, {}, {})'.format(self.in_channels, self.out_channels, self.act.__name__)
