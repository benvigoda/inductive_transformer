from torch import nn  # type: ignore


class EncoderBernoulliCategorical(nn.Module):

    def __init__(self, hyperparams):
        super(EncoderBernoulliCategorical, self).__init__()
        self.hyperparams = hyperparams

    def forward(self, u):

        v[0] = u[1][0]/u[0][0]
        v[1] = u[1][1]/u[0][1]

        return v
