import torch  # type: ignore
from torch import nn  # type: ignore


class DecoderUniverse(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderUniverse, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

        self.z = None

    def forward(self, u):
        z = torch.empty((2, self.hyperparams.layer_width))

        # u[heads/tails][below_lw?][above_lw?]
        # left

        z[0][0] = u[0][0][0] * u[0][0][1]  # all the others
        z[1][0] = u[1][0][0] * u[1][0][1] + u[1][0][0] * u[0][0][1] + u[0][0][0] * u[1][0][1]
        # right
        z[0][1] = u[0][1][0] * u[0][1][1]
        z[1][1] = u[1][1][0] * u[1][1][1] + u[1][1][0] * u[0][1][1] + u[0][1][0] * u[1][1][1]

        z = nn.functional.normalize(z, p=1, dim=0)
        self.z = z
        return z
