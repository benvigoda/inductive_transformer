import torch  # type: ignore
from torch import nn  # type: ignore
from torch_transformer.helper_functions import custom_normalize


class DecoderBernoulliCategorical(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderBernoulliCategorical, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

        self.categorical = None

    def forward(self, bernoulli):

        # bernoulli is size (2, layer_width)
        assert bernoulli.shape == (2, self.hyperparams.layer_width)

        # categorical is size = (1, layer_width)
        categorical = torch.empty((1, self.hyperparams.layer_width), device=bernoulli.device)

        categorical[0] = bernoulli[1] / (bernoulli[0] + 1e-9)

        # categorical = nn.functional.normalize(categorical, p=1, dim=1)
        categorical = custom_normalize(categorical, dim=1)

        self.categorical = categorical

        return categorical
