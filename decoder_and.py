from torch import nn  # type: ignore


class DecoderAnd(nn.Module):

    def __init__(self, hyperparams):
        super(DecoderAnd, self).__init__()
        self.hyperparams = hyperparams

    def forward(self, z):

        # y[1][0] = z[1][0]*y[1][0]
        # y[0][0] = z[0][0]*y[1][0] + x[1][0]*y[0][0] + x[0][0]*y[0][0]

        # y[1][1] = x[1][1]*y[1][1]
        # y[0][1] = x[0][1]*y[1][1] + x[1][1]*y[0][1] + x[0][1]*y[0][1]

        x = torch.normalize(x, p=1, dim=0)
        y = torch.normalize(y, p=1, dim=0)

        return x, y
