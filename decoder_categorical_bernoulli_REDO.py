import torch  # type: ignore
from torch import nn  # type: ignore


class DecoderCategoricalBernoulli(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderCategoricalBernoulli, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

    def forward(self, v):

        # v[below_lw][above_lw]
        # u[heads/tails][below_lw][above_lw]

        # there are four signals coming down.  
        # the two going to the left below are:
        # v[0][0]
        # v[0][1]

        # and two going to the right below:
        # v[1][0]
        # v[1][1]

        # we need to convert all of these to bernoullis
        # left:
        u[1][0][0] = v[0][0]
        u[0][0][0] = v[1][0] # for the tails state take the value that v[0][0] is normalized against which is v[1][0]

        u[1][0][1] = v[0][1]
        u[0][0][1] = v[1][1]

        # right:
        u[1][1][0] = v[1][0] # the last two indices on the left hand side are equal to the indices on the right hand side
        u[0][1][0] = v[0][0]

        u[1][1][1] = v[1][1] # the last two indices on the left hand side are equal to the indices on the right hand side
        u[0][1][1] = v[0][1]

        # FIXME: this will not normalize each of the heads/tails signals separately
        u[:,:,0] = torch.normalize(u[:,:,0], p=1, dim=0)
        u[:,:,1] = torch.normalize(u[:,:,1], p=1, dim=0)
        # what we should do instead is:

        return u