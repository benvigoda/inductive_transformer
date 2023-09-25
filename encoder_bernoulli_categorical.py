import torch  # type: ignore
from torch import nn  # type: ignore
from helper_functions import custom_normalize


class EncoderBernoulliCategorical(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderBernoulliCategorical, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

        self.v = None

    def forward(self, u):
        v = torch.empty((2, 2))

        # there's four coins coming in
        # to convert coins to categorical, it's always head divided by tails
        # and then normalize the categoricals
        # v[below_lw][above_lw] = u[heads][below_lw][above_lw] / u[tails][below_lw][above_lw]
        v[0][0] = u[1][0][0]/u[0][0][0]  # straight up the left edge
        v[0][1] = u[1][0][1]/u[0][0][1]  # the left universe to the right pi_a cross connection

        v[1][0] = u[1][1][0]/u[0][1][0]  # the right universe to the left pi_a cross connection
        v[1][1] = u[1][1][1]/u[0][1][1]  # straight up on the right

        # we want to normalize is the inputs to a specific pi_a, remember from the encoder universe factor:
        # v[0][0] + v[1][0] = 1
        # v[0][1] + v[1][1] = 1
        # v = nn.functional.normalize(v, p=1, dim=0)
        v = custom_normalize(v, dim=0)
        self.v = v
        return v
