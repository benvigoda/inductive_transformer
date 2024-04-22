import torch  # type: ignore
from torch import nn  # type: ignore
from inductive_transformer.torch_transformer.helper_functions import custom_normalize


class DecoderPositionPi(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderPositionPi, self).__init__()
        self.hyperparams = hyperparams
        self.num_positions = self.hyperparams.num_positions
        self.layer_width = self.hyperparams.layer_width
        self.active_layer = active_layer
        if hyperparams.decoder_position_pi_weights is not None:
            initial_weights = hyperparams.decoder_position_pi_weights[active_layer]
            if hyperparams.init_perturb_weights:
                random_noise = torch.randn(self.num_positions, self.layer_width) * 0.1
                self.weights = nn.Parameter(
                    torch.zeros(self.num_positions, self.layer_width) + initial_weights + random_noise,
                    requires_grad=True
                )
            else:
                self.weights = initial_weights
        else:
            self.weights = nn.Parameter(torch.ones(self.num_positions, self.layer_width), requires_grad=True)
            nn.init.normal_(self.weights, mean=1, std=0.1)
        self.relu = nn.ReLU()

        self.rho = None
        self.t = None

    def forward(self, x):
        self.x = x
        # we expect x to be already normalized categorical

        prob_weights = self.relu(self.weights) + 1e-9

        # we are going to output a categorical distribution over tokens at every lw in the layer
        # each of these output categoricals will be of length vocab_size
        # each categorical will be normalized, not to 1, but to the x value at this lw
        # an easy way to do this is to normalize the prob weights in advance in dim=0
        prob_weights = custom_normalize(prob_weights, dim=0)  # FIXME: could be causing problems

        # and then since x comes in as categorical of size (1, layer_width)
        assert x.shape == (1, self.layer_width)

        x = custom_normalize(x, dim=1)
        # we want to stack x in dim = 0
        x_stacked = torch.cat([x for vs in range(self.num_positions)], dim=0)

        # element-wise product of weight tensor and y_stacked
        rho = prob_weights * x_stacked
        assert rho.shape == (self.num_positions, self.layer_width)

        self.rho = rho
        return rho
