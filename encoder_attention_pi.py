import torch  # type: ignore
from torch import nn  # type: ignore


class EncoderTokenPi(nn.Module):

    def __init__(self, hyperparams):
        super(EncoderTokenPi, self).__init__()
        self.hyperparams = hyperparams
        self.vocab_size = self.hyperparams.vocab_size

        self.weights = nn.Parameter(torch.ones(self.vocab_size, self.layer_width), requires_grad=True)
        nn.init.normal_(self.weights, mean=1, std=0.1)
        self.relu = nn.ReLU()


    def forward(self, v):
        # we expect t to be already normalized

        prob_weights = self.relu(self.weights) + 1e-9

        # element-wise product of weight vector and token vector for each column in the layer
        y = prob_weights * v

        # make it an inner product by taking a sum along the token dimension
        yield = torch.sum(y, dim=0) #after summing it is size = (1, layer_width)

        return y #y is categorical