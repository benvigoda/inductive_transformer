from torch import nn  # type: ignore


class EncoderCategoricalBernoulli(nn.Module):

    def __init__(self, hyperparams):
        super(EncoderCategoricalBernoulli, self).__init__()
        self.hyperparams = hyperparams

    def forward(self, categorical):
        # categorical is size = (1, layer_width)
        # bernoulli is size (2, layer_width)
        bernoulli[1][0] = categorical[0, 0]
        bernoulli[0][0] = categorical[0, 1]

        bernoulli[1][1] = categorical[0, 1]
        bernoulli[0][1] = categorical[0, 0]

        bernoulli = torch.normalize(bernoulli, p=1, dim=0)

        return bernoulli
