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

    def forward(self, t):
        prob_weights = self.relu(self.weights) + 1e-9

        x[:,1] = prob_weights * t[:,1]
        p_y1 = torch.sum(x, dim=0)

        return x
