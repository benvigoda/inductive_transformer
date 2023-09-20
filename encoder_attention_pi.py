import torch  # type: ignore
from torch import nn  # type: ignore


class EncoderAttentionPi(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderAttentionPi, self).__init__()
        self.hyperparams = hyperparams
        self.vocab_size = self.hyperparams.vocab_size
        self.active_layer = active_layer

        if hyperparams.encoder_attention_pi_weights is not None:
            self.weights = hyperparams.encoder_attention_pi_weights[active_layer]
        else:
            self.weights = nn.Parameter(torch.ones(self.layer_width, self.layer_width), requires_grad=True)
            nn.init.normal_(self.weights, mean=1, std=0.1)
        self.relu = nn.ReLU()

    def forward(self, v):
        # we expect v to be already normalized categorical

        prob_weights = self.relu(self.weights) + 1e-9

        # element-wise product of weight vector and token vector for each column in the layer
        y = prob_weights * v

        # make it an inner product by taking a sum along the token dimension
        y = torch.sum(y, dim=0)  # after summing it is size = (1, layer_width)
        assert y.shape == (1, self.layer_width)

        return y  # y is categorical
