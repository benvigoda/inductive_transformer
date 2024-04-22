import torch  # type: ignore
from torch import nn  # type: ignore
from torch_transformer.helper_functions import custom_normalize


'''
Let vocab_size = 4, num_positions = 3, and layer_width = 2

The data is then a tensor that is size (num_positions=3, vocab size=4)

There is a Forney equals gates at each specific (position and word)

The left column has three pi_t's, each with 4 vocab words that can explain away the entire data
The right column also has this.

When the data says "small dog", then we want the left column to explain away the data

When the data says "big cat", then we want the right column to explain away the data

The entire layer_width must participate in explaining away the changing data.

We will need an open closed universe

... maybe we can do this in a simpler way that will not require the open closed universe:
in encoder_layer.py clone the data and send one clone straight up and one clone across
'''


class EncoderPositionPi(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderPositionPi, self).__init__()
        self.hyperparams = hyperparams
        self.num_positions = self.hyperparams.num_positions
        self.layer_width = self.hyperparams.layer_width
        self.active_layer = active_layer

        if hyperparams.encoder_position_pi_weights is not None:
            initial_weights = hyperparams.encoder_position_pi_weights[active_layer]
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
        self.x = None

    def forward(self, rho):
        assert rho.shape == (self.num_positions, self.layer_width)
        # we need to normalize rho
        # rho = custom_normalize(rho, dim=0)
        self.rho = rho

        prob_weights = self.relu(self.weights) + 1e-9  # FIXME? Do we need the 1e-9?
        # NOTE: we decided to normalize the weights (it shouldn't matter)
        prob_weights = custom_normalize(prob_weights, dim=0)

        # element-wise product of weight vector and token vector for each column in the layer
        x = prob_weights * rho

        # make it an inner product by taking a sum along the token dimension
        x = torch.sum(x, dim=0, keepdim=True)  # after summing it is size = (1, layer_width)
        x = custom_normalize(x, dim=1)
        self.x = x
        return x  # x is categorical
