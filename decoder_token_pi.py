import torch  # type: ignore
from torch import nn  # type: ignore
from helper_functions import custom_normalize


class DecoderTokenPi(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderTokenPi, self).__init__()
        self.hyperparams = hyperparams
        self.num_positions = self.hyperparams.num_positions
        self.layer_width = self.hyperparams.layer_width
        self.vocab_size = self.hyperparams.vocab_size
        self.active_layer = active_layer
        if hyperparams.decoder_token_pi_weights is not None:
            self.weights = hyperparams.decoder_token_pi_weights[active_layer]
        else:
            self.weights = nn.Parameter(torch.ones(self.num_positions, self.vocab_size, self.layer_width), requires_grad=True)
            nn.init.normal_(self.weights, mean=1, std=0.1)
        self.relu = nn.ReLU()

        self.rho = None
        self.t = None

    def forward(self, rho):
        self.rho = rho
        assert rho.shape == (self.num_positions, self.layer_width)
        # we expect rho to be already normalized categorical

        prob_weights = self.relu(self.weights) + 1e-9

        # we are going to output a categorical distribution over tokens at every lw in the layer
        # each of these output categoricals will be of length vocab_size
        # each categorical will be normalized, not to 1, but to the x value at this lw
        # an easy way to do this is to normalize the prob weights in advance in dim=0
        # prob_weights = nn.functional.normalize(prob_weights, p=1, dim=0)
        # prob_weights = custom_normalize(prob_weights, dim=0)

        # rho = custom_normalize(rho, dim=1)  #FIXME: do we want this? We already do it in the decoder_position_pi

        # we want to stack x in dim = 0
        rho_stacked = torch.stack([rho for vs in range(self.vocab_size)], dim=1)

        # element-wise product of weight tensor and y_stacked
        t = prob_weights * rho_stacked
        assert t.shape == (self.num_positions, self.vocab_size, self.layer_width)

        # t = custom_normalize(t, dim=1)

        self.t = t
        return t
