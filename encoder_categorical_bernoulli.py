import torch  # type: ignore
from torch import nn  # type: ignore
from helper_functions import custom_normalize


class EncoderCategoricalBernoulli(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderCategoricalBernoulli, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

        self.bernoulli = None

    def forward(self, categorical):
        # categorical is size = (1, layer_width)
        assert categorical.shape == (1, self.hyperparams.layer_width)
        # bernoulli is size (2, layer_width)
        bernoulli = torch.empty((2, self.hyperparams.layer_width))

        # we can ignore the dim=0 index in the categorical. It is always= 0.
        # prob of bernoulli = 1 on left side == prob of categorical on left side:
        bernoulli[1][0] = categorical[0, 0]
        # prob of bernoulli = 0 on left side is just all the other probability mass in the categorical:
        bernoulli[0][0] = categorical[0, 1]

        bernoulli[1][1] = categorical[0, 1]
        bernoulli[0][1] = categorical[0, 0]

        # bernoulli = nn.functional.normalize(bernoulli, p=1, dim=0)
        bernoulli = custom_normalize(bernoulli, dim=0)
        self.bernoulli = bernoulli
        return bernoulli
