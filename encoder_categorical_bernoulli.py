import torch  # type: ignore
from torch import nn  # type: ignore


class EncoderCategoricalBernoulli(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderCategoricalBernoulli, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

    def forward(self, categorical):
        # categorical is size = (1, layer_width)
        assert categorical.shape == (1, self.hyperparams.layer_width)
        # bernoulli is size (2, layer_width)
        bernoulli = torch.empty((2, self.hyperparams.layer_width))

        bernoulli[1][0] = categorical[0, 0]
        bernoulli[0][0] = categorical[0, 1]

        bernoulli[1][1] = categorical[0, 1]
        bernoulli[0][1] = categorical[0, 0]

        bernoulli = torch.normalize(bernoulli, p=1, dim=0)

        return bernoulli
