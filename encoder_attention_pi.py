import torch  # type: ignore
from torch import nn  # type: ignore
from helper_functions import custom_normalize


class EncoderAttentionPi(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderAttentionPi, self).__init__()
        self.hyperparams = hyperparams
        self.vocab_size = self.hyperparams.vocab_size
        self.layer_width = self.hyperparams.layer_width
        self.active_layer = active_layer

        if hyperparams.encoder_attention_pi_weights is not None:
            self.weights = hyperparams.encoder_attention_pi_weights[active_layer]
        else:
            if self.active_layer == 0:
                self.weights = nn.Parameter(torch.ones(self.layer_width, self.layer_width), requires_grad=True)
                nn.init.normal_(self.weights, mean=1, std=0.1)
            else:
                self.weights = nn.Parameter(torch.eye(self.layer_width), requires_grad=True)
                self.weights.data += torch.randn_like(self.weights) * 0.1
        self.relu = nn.ReLU()

        self.y = None

    def forward(self, v):
        assert v.shape == (self.layer_width, self.layer_width)
        # we expect v to be already normalized categorical

        prob_weights = self.relu(self.weights) + 1e-9
        # NOTE: we decided not to normalize the weights (it shouldn't matter)
        # prob_weights = nn.functional.normalize(prob_weights, p=1, dim=0)
        prob_weights = custom_normalize(prob_weights, dim=0)

        # element-wise product of weight vector and token vector for each column in the layer
        y = prob_weights * v

        # make it an inner product by taking a sum along the choice dimension
        y = torch.sum(y, dim=0, keepdim=True)  # after summing it is size = (1, layer_width)
        assert y.shape == (1, self.layer_width)

        # y = nn.functional.normalize(y, p=1, dim=1)
        y = custom_normalize(y, dim=1)
        self.y = y
        return y  # y is categorical
