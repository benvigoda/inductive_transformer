import torch  # type: ignore
from torch import nn  # type: ignore


class EncoderBernoulliCategorical(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderBernoulliCategorical, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

    def forward(self, u):

        # FIXME
        v = torch.empty((2, ))

        v[0] = u[1][0]/u[0][0]
        v[1] = u[1][1]/u[0][1]

        return v
