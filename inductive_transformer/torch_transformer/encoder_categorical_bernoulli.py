import torch  # type: ignore
from torch import nn  # type: ignore
from inductive_transformer.torch_transformer.helper_functions import custom_normalize


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
        bernoulli = torch.empty(
            (2, self.hyperparams.layer_width), device=categorical.device
        )

        # The probability of a bernoulli variable being true is the same as the probability of the
        # corresponding categorical state.
        bernoulli[1] = categorical

        # The probability of a bernoulli variable being false is the sum of the probabilities of all
        # the other categorical states.
        # Note: if categorical[i][j] is much larger than categorical[i][k] for k != j, then this
        # method of performing the calculation introduces a lot of rounding error.
        bernoulli[0] = categorical.sum(dim=-1, keepdim=True) - categorical

        # bernoulli = nn.functional.normalize(bernoulli, p=1, dim=0)
        bernoulli = custom_normalize(bernoulli, dim=0)
        self.bernoulli = bernoulli
        return bernoulli
