import torch  # type: ignore
from torch import nn  # type: ignore


class EncoderUniverse(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderUniverse, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

    def forward(self, z):
        # dim=0 indexes the state of Bernoulli, 0 or 1
        # dim=1 indexes the layer_width

        # u[0][1] = p(parent=0) right
        # z[0][left] = p(child=1)[left]
        # u[1][0] = z[1][0]*u[0][1] + z[1][0]*u[1][1]
        # u[0][0] = z[1][0]*u[1][1] + z[0][0]*u[0][1]
        # u[1][1] = z[1][1]*u[0][0] + z[1][1]*u[1][0]
        # u[0][1] = z[1][1]*u[1][0] + z[0][1]*u[0][0]

        # u[1][0] = z[1][0]*0.5 + z[1][0]*0.5
        # u[0][0] = z[1][0]*0.5 + z[0][0]*0.5
        # u[1][1] = z[1][1]*0.5 + z[1][1]*0.5
        # u[0][1] = z[1][1]*0.5 + z[0][1]*0.5

        # FIXME
        u = torch.empty(self.hyperparams.layer_width, self.hyperparams.layer_width)

        u[1][0] = z[1][0]
        u[0][0] = 0.5

        u[1][1] = z[1][1]
        u[0][1] = 0.5

        torch.normalize(u, p=1, dim=0)
        return u
