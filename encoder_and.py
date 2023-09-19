from torch import nn  # type: ignore


class EncoderAnd(nn.Module):

    def __init__(self, hyperparams):
        super(EncoderAnd, self).__init__()
        self.hyperparams = hyperparams

    def forward(self, x, y):

        z = x + y # FIXME
        
        return z
