import torch  # type: ignore
from torch import nn  # type: ignore


class EncoderTokenPi(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderTokenPi, self).__init__()
        self.hyperparams = hyperparams
        self.vocab_size = self.hyperparams.vocab_size
        self.active_layer = active_layer

        if hyperparams.encoder_token_pi_weights is not None:
            self.weights = hyperparams.encoder_token_pi_weights[active_layer]
        else:
            self.weights = nn.Parameter(torch.ones(self.vocab_size, self.layer_width), requires_grad=True)
            nn.init.normal_(self.weights, mean=1, std=0.1)
        self.relu = nn.ReLU()

    def forward(self, t):
        # we expect t to be already normalized

        prob_weights = self.relu(self.weights) + 1e-9

        # element-wise product of weight vector and token vector for each column in the layer
        x = prob_weights * t

        # make it an inner product by taking a sum along the token dimension
        x = torch.sum(x, dim=0)  # after summing it is size = (1, layer_width)

        return x  # x is categorical
