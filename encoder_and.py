import torch  # type: ignore
from torch import nn  # type: ignore
from helper_functions import custom_normalize


class EncoderAnd(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderAnd, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

        self.x = None
        self.y = None
        self.z = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        z = torch.empty((2, 2), device=x.device)  # size = (2, layer_width)
        # left
        z[1][0] = x[1][0]*y[1][0]  # 1 * 0.5 = 0.5
        z[0][0] = x[0][0]*y[1][0] + x[1][0]*y[0][0] + x[0][0]*y[0][0]  # 0 + 1 * 0.5 + 0 = 0.5

        z[1][1] = x[1][1]*y[1][1]  # 0 * 0.5 = 0
        z[0][1] = x[0][1]*y[1][1] + x[1][1]*y[0][1] + x[0][1]*y[0][1]  # 1 * 0.5 + 0 + 0 = 0

        # z = nn.functional.normalize(z, p=1, dim=0)
        z = custom_normalize(z, dim=0)
        self.z = z
        return z
