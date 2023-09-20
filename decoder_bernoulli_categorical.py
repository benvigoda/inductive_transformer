import torch  # type: ignore
from torch import nn  # type: ignore


class DecoderBernoulliCategorical(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderBernoulliCategorical, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

    def forward(self, bernoulli):

        # bernoulli is size (2, layer_width)
        assert bernoulli.shape == (2, self.hyperparams.layer_width)

        # categorical is size = (1, layer_width)
        categorical = torch.empty((1, self.hyperparams.layer_width))

        categorical[0, 0] = bernoulli[1][0]/bernoulli[0][0]
        categorical[0, 1] = bernoulli[1][1]/bernoulli[0][1]

        categorical = torch.normalize(categorical, p=1, dim=1)

        return categorical
