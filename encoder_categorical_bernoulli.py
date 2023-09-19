from torch import nn  # type: ignore


class EncoderCategoricalBernoulli(nn.Module):

    def __init__(self, hyperparams):
        super(EncoderCategoricalBernoulli, self).__init__()
        self.hyperparams = hyperparams

    def forward(self, categorical):

        bernoulli = categorical # FIXME
        
        return bernoulli
