import torch  # type: ignore
from torch import nn  # type: ignore
from helper_functions import custom_normalize


class DecoderAttentionPi(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderAttentionPi, self).__init__()
        self.hyperparams = hyperparams
        self.layer_width = self.hyperparams.layer_width
        self.active_layer = active_layer
        if hyperparams.decoder_attention_pi_weights is not None:
            initial_weights = hyperparams.decoder_attention_pi_weights[active_layer]
            if hyperparams.init_perturb_weights:
                random_noise = torch.randn(self.layer_width, self.layer_width) * 0.1
                self.weights = nn.Parameter(
                    torch.zeros(self.layer_width, self.layer_width) + initial_weights + random_noise,
                    requires_grad=True
                )
            else:
                self.weights = initial_weights
        else:
            # self.weights[below_lw][above_lw], where below in the decoder is towards the output, and above is from the encoder
            if self.active_layer == 0:
                self.weights = nn.Parameter(torch.ones(self.layer_width, self.layer_width), requires_grad=True)
                nn.init.normal_(self.weights, mean=1, std=0.1)
            else:
                self.weights = nn.Parameter(torch.eye(self.layer_width), requires_grad=True)
                self.weights.data += torch.randn_like(self.weights) * 0.1
        self.relu = nn.ReLU()

        self.y = None
        self.v = None

    def forward(self, y):
        self.y = y
        # we expect y to be already normalized categorical

        prob_weights = self.relu(self.weights) + 1e-9

        # we are going to output a categorical distribution over tokens at every lw in the layer
        # each of these output categoricals will be of length vocab_size
        # each categorical will be normalized, not to 1, but to the y value at this lw
        # an easy way to do this is to normalize the prob weights in advance in dim=0
        # prob_weights = nn.functional.normalize(prob_weights, p=1, dim=0)
        prob_weights = custom_normalize(prob_weights, dim=1)

        # and then since y comes in as categorical of size (1, layer_width)
        assert y.shape == (1, self.layer_width)
        # y = nn.functional.normalize(y, p=1, dim=1)
        y = custom_normalize(y, dim=1)
        # we want to stack x in dim = 0
        y_stacked = torch.cat([y for lw in range(self.layer_width)], dim=0)

        # element-wise product of weight tensor and y_stacked
        v = prob_weights * y_stacked
        assert v.shape == (self.layer_width, self.layer_width)

        self.v = v
        return v
