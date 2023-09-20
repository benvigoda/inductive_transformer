import torch  # type: ignore
from torch import nn  # type: ignore


class DecoderTokenPi(nn.Module):

    def __init__(self, hyperparams):
        super(DecoderTokenPi, self).__init__()
        self.hyperparams = hyperparams
        self.layer_width = self.hyperparams.layer_width
        self.vocab_size = self.hyperparams.vocab_size

        self.weights = nn.Parameter(torch.ones(self.vocab_size, self.layer_width), requires_grad=True)
        nn.init.normal_(self.weights, mean=1, std=0.1)
        self.relu = nn.ReLU()


    def forward(self, x):
        # we expect y to be already normalized categorical

        prob_weights = self.relu(self.weights) + 1e-9

        # element-wise product of weight vector and token vector for each column in the layer
        for y in range(self.vocab_size):
            y_stacked = torch.stack([y_stacked], y, dim=0)
        t = prob_weights * y_stacked

        t = torch.normalize(t, p=1, dim=0)
        # we need to make sure we sample words from dim=0 and dim=1 in v correctly
        # to sample a word in a single position, we use an open closed universe to convert
        # to a closed universe of positions

        return t #t is categorical on dim=0