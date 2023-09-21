import torch  # type: ignore
from torch import nn  # type: ignore


class EncoderUniverse(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderUniverse, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

    def forward(self, z):

        # FIXME
        u = torch.empty(self.hyperparams.layer_width, self.hyperparams.layer_width)

        # if this parent=1, then child=1, the other_parent can be whatever
        # child=1 and other_parent=1
        # child=1 and other_parent=0

        # if this parent=0, then child=other_parent
        # child=1 and other_parent=1
        # child=0 and other_parent=0

        # z[1][0] = 1 - z[0][0] is the left child
        # z[1][1] = 1 - z[0][1] is the right child

        # z[heads/tails][below_lw]
        # u[heads/tails][below_lw][above_lw]
        # so when we are considering parent u[a][b][c] we are using child z[][b]
        
        '''
        # FOUR COINS:

        # p(parent_a = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][0][0] = z[1][0] * 0.5 + z[1][0] * 0.5
        # p(parent_a = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][0][0] = z[][] * 0.5 + z[1][0] * 0.5

        # p(parent_b = 1) = p(child = 1)p(parent_a = 0) + p(child = 1)p(parent_a = 1)
        u[1][0][1] = z[][] * 0.5 + z[1][0] * 0.5
        # p(parent_b = 0) = p(child = 1)p(parent_a = 1) + p(child = 0)p(parent_a = 0)
        u[0][0][1] = z[][] * 0.5 + z[1][0] * 0.5


        # p(parent_a = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][1][0] = z[1][0] * 0.5 + z[1][0] * 0.5
        # p(parent_a = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][1][0]= z[][] * 0.5 + z[1][0] * 0.5

        # p(parent_b = 1) = p(child = 1)p(parent_a = 0) + p(child = 1)p(parent_a = 1)
        u[1][1][1] = z[][] * 0.5 + z[1][0] * 0.5
        # p(parent_b = 0) = p(child = 1)p(parent_a = 1) + p(child = 0)p(parent_a = 0)
        u[0][1][1] = z[][] * 0.5 + z[1][0] * 0.5

        But all of the other parents are going to be 0.5 anyway!
        So we're gonna have:

        # p(parent_a = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][0][0] = z[][] * 0.5 + z[][] * 0.5
        # p(parent_a = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][0][0] = z[][] * 0.5 + z[][] * 0.5

        # p(parent_b = 1) = p(child = 1)p(parent_a = 0) + p(child = 1)p(parent_a = 1)
        u[1][0][1] = z[][] * 0.5 + z[1][0] * 0.5
        # p(parent_b = 0) = p(child = 1)p(parent_a = 1) + p(child = 0)p(parent_a = 0)
        u[0][0][1] = z[][] * 0.5 + z[1][0] * 0.5


        # p(parent_a = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][1][0] = z[1][0] * 0.5 + z[1][0] * 0.5
        # p(parent_a = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][1][0]= z[][] * 0.5 + z[1][0] * 0.5

        # p(parent_b = 1) = p(child = 1)p(parent_a = 0) + p(child = 1)p(parent_a = 1)
        u[1][1][1] = z[][] * 0.5 + z[1][0] * 0.5
        # p(parent_b = 0) = p(child = 1)p(parent_a = 1) + p(child = 0)p(parent_a = 0)
        u[0][1][1] = z[][] * 0.5 + z[1][0] * 0.5

        

        so we just need to get the child indices right
        the first index of z[heads/tails] so you know that

        # p(parent_a = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][0][0] = z[1][] * 0.5 + z[1][] * 0.5
        # p(parent_a = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][0][0] = z[1][] * 0.5 + z[0][] * 0.5

        # p(parent_b = 1) = p(child = 1)p(parent_a = 0) + p(child = 1)p(parent_a = 1)
        u[1][0][1] = z[1][] * 0.5 + z[1][] * 0.5
        # p(parent_b = 0) = p(child = 1)p(parent_a = 1) + p(child = 0)p(parent_a = 0)
        u[0][0][1] = z[1][] * 0.5 + z[0][0] * 0.5


        # p(parent_a = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][1][0] = z[1][] * 0.5 + z[1][] * 0.5
        # p(parent_a = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][1][0]= z[1][] * 0.5 + z[0][] * 0.5

        # p(parent_b = 1) = p(child = 1)p(parent_a = 0) + p(child = 1)p(parent_a = 1)
        u[1][1][1] = z[1][] * 0.5 + z[1][] * 0.5
        # p(parent_b = 0) = p(child = 1)p(parent_a = 1) + p(child = 0)p(parent_a = 0)
        u[0][1][1] = z[1][] * 0.5 + z[0][] * 0.5


        so we just need to get the last child indices right
        the second index of the child is left/right
        # TODO 
        same index as the parent indices for left/right which is the middle index dim=1?

        # p(parent_a = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][0][0] = z[1][] * 0.5 + z[1][] * 0.5
        # p(parent_a = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][0][0] = z[1][] * 0.5 + z[0][] * 0.5

        # p(parent_b = 1) = p(child = 1)p(parent_a = 0) + p(child = 1)p(parent_a = 1)
        u[1][0][1] = z[1][] * 0.5 + z[1][] * 0.5
        # p(parent_b = 0) = p(child = 1)p(parent_a = 1) + p(child = 0)p(parent_a = 0)
        u[0][0][1] = z[1][] * 0.5 + z[0][0] * 0.5


        # p(parent_a = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][1][0] = z[1][] * 0.5 + z[1][] * 0.5
        # p(parent_a = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][1][0]= z[1][] * 0.5 + z[0][] * 0.5

        # p(parent_b = 1) = p(child = 1)p(parent_a = 0) + p(child = 1)p(parent_a = 1)
        u[1][1][1] = z[1][] * 0.5 + z[1][] * 0.5
        # p(parent_b = 0) = p(child = 1)p(parent_a = 1) + p(child = 0)p(parent_a = 0)
        u[0][1][1] = z[1][] * 0.5 + z[0][] * 0.5

        '''
        # let's look at the right-side encoder bernoulli-categorical
        # the left parent of the this factor cross over to the left side
        # the right parent goes straight up to the pi_a above

        # let's look at the left-side encoder bernoulli-categorical
        # the right parent of the this factor cross over to the right side
        # the left parent goes straight up to the pi_a above

        # and they both need to be dice at this point where
        # all that means is that the two signals goiing to the left pi_a need to sum to 1
        # and similarly for the two signals going to the right pi_a also need to sum to 1

        u = torch.normalize(u, p=1, dim=0)
        return u

        # encoder_bernoulli_categorical.py
        # there's four coins coming in
        # to convert coins to categorical, it's always head divided by tails
        # and then normalize the categoricals
        # v[below_lw][above_lw] = u[heads][below_lw][above_lw] / u[tails][below_lw][above_lw]
        v[0][0] = u[1][0][0]/u[0][0][0]  # straight up the left edge
        v[0][1] = u[1][0][1]/u[0][0][1]  # the left universe to the right pi_a cross connection

        v[1][0] = u[1][1][0]/u[0][1][0]  # the right universe to the left pi_a cross connection
        v[1][1] = u[1][1][1]/u[0][1][1]  # straight up on the right

        # we want to normalize is the inputs to a specific pi_a
        v = torch.normalize(v, p=1, dim=0)
