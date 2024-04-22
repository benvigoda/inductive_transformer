import torch  # type: ignore
from torch import nn  # type: ignore
from inductive_transformer.torch_transformer.helper_functions import custom_normalize


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
        z = torch.empty((2, self.hyperparams.layer_width), device=x.device)
        z[1] = x[1] * y[1]
        z[0] = x[0] * y[1] + x[1] * y[0] + x[0] * y[0]

        # z = nn.functional.normalize(z, p=1, dim=0)
        z = custom_normalize(z, dim=0)
        self.z = z
        return z
