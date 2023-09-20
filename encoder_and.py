import torch  # type: ignore
from torch import nn  # type: ignore


class EncoderAnd(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderAnd, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

    def forward(self, x, y):

        z = torch.empty((2, 2))
        z[1][0] = x[1][0]*y[1][0]
        z[0][0] = x[0][0]*y[1][0] + x[1][0]*y[0][0] + x[0][0]*y[0][0]

        z[1][1] = x[1][1]*y[1][1]
        z[0][1] = x[0][1]*y[1][1] + x[1][1]*y[0][1] + x[0][1]*y[0][1]

        z_normed = torch.normalize(z, p=1, dim=0)

        return z_normed
