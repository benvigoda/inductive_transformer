import torch  # type: ignore
from torch import nn  # type: ignore
from helper_functions import custom_normalize


class EncoderTokenPi(nn.Module):

    # We have a token pi for each position pi, so we'll make num_positions clones of the token pi
    def __init__(self, hyperparams, active_layer: int):
        super(EncoderTokenPi, self).__init__()
        self.hyperparams = hyperparams
        self.num_positions = self.hyperparams.num_positions
        self.vocab_size = self.hyperparams.vocab_size
        self.layer_width = self.hyperparams.layer_width
        self.active_layer = active_layer

        if hyperparams.encoder_token_pi_weights is not None:
            self.weights = hyperparams.encoder_token_pi_weights[active_layer]
        else:
            self.weights = nn.Parameter(torch.ones(self.num_positions, self.vocab_size, self.layer_width), requires_grad=True)
            nn.init.normal_(self.weights, mean=1, std=0.1)
        self.relu = nn.ReLU()

        self.t = None
        self.rho = None

    def forward(self, t):
        self.t = t
        assert t.shape == (self.num_positions, self.vocab_size, self.layer_width)
        # we expect t to be already normalized

        prob_weights = self.relu(self.weights) + 1e-9
        # NOTE: we decided not to normalize the weights (it shouldn't matter)
        # prob_weights = nn.functional.normalize(prob_weights, p=1, dim=0)
        # prob_weights = custom_normalize(prob_weights, dim=1)

        # element-wise product of weight vector and token vector for each column in the layer
        rho = prob_weights * t

        # make it an inner product by taking a sum along the token dimension
        rho = torch.sum(rho, dim=1, keepdim=True)

        # after summing it is size = (num_positions, 1, layer_width)
        rho = rho.squeeze(dim=1)
        # and now it will just be size = (num_positions, layer_width)
        # rho = custom_normalize(rho, dim=1)

        self.rho = rho
        return rho  # rho is categorical
