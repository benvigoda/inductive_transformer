import torch  # type: ignore
from torch import nn  # type: ignore


class DecoderTokenPi(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderTokenPi, self).__init__()
        self.hyperparams = hyperparams
        self.layer_width = self.hyperparams.layer_width
        self.vocab_size = self.hyperparams.vocab_size
        self.active_layer = active_layer
        if hyperparams.decoder_token_pi_weights is not None:
            self.weights = hyperparams.decoder_token_pi_weights[active_layer]
        else:
            self.weights = nn.Parameter(torch.ones(self.vocab_size, self.layer_width), requires_grad=True)
            nn.init.normal_(self.weights, mean=1, std=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # we expect x to be already normalized categorical

        prob_weights = self.relu(self.weights) + 1e-9

        # we are going to output a categorical distribution over tokens at every lw in the layer
        # each of these output categoricals will be of length vocab_size
        # each categorical will be normalized, not to 1, but to the x value at this lw
        # an easy way to do this is to normalize the prob weights in advance in dim=0
        prob_weights = nn.functional.normalize(prob_weights, p=1, dim=0)

        # and then since x comes in as categorical of size (1, layer_width)
        assert x.shape == (1, self.layer_width)
        x = nn.functional.normalize(x, p=1, dim=1)
        # we want to stack x in dim = 0
        x_stacked = torch.cat([x for vs in range(self.vocab_size)], dim=0)

        # element-wise product of weight tensor and y_stacked
        t = prob_weights * x_stacked
        assert t.shape == (self.vocab_size, self.layer_width)

        return t
