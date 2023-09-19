from torch import nn  # type: ignore


class EncoderBernoulliCategorical(nn.Module):

    def __init__(self, hyperparams):
        super(EncoderBernoulliCategorical, self).__init__()
        self.hyperparams = hyperparams

    def forward(self, u):

        v = u
        return v
