import torch  # type: ignore
from torch import nn  # type: ignore
from inductive_transformer.torch_transformer.helper_functions import custom_normalize


class EncoderBernoulliCategorical(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderBernoulliCategorical, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

        self.v = None

    def forward(self, u):
        v = torch.empty((self.hyperparams.layer_width, self.hyperparams.layer_width), device=u.device)

        # there's four coins coming in
        # to convert coins to categorical, it's always head divided by tails
        # and then normalize the categoricals
        # v[below_lw][above_lw] = u[heads][below_lw][above_lw] / u[tails][below_lw][above_lw]
        v = u[1] / (u[0] + 1e-9)

        # we want to normalize is the inputs to a specific pi_a, remember from the encoder universe factor:
        # v[0][0] + v[1][0] = 1
        # v[0][1] + v[1][1] = 1
        # v = nn.functional.normalize(v, p=1, dim=0)
        v = custom_normalize(v, dim=0)
        self.v = v
        return v
