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

        # we are developing a closed to open universe encoder factor
        # it's output will go into a bernoulli to categorical encoder factor

        # we are going to think about what output this universe factor needs to produce
        # in terms of the inputs it receives from below

        # the inputs below are z's and they live in a closed universe with a limited layer width
        # the outputs go to the pi_a's above who live in an open universe of expanded choices

        # in the decoder direction, the pi_a's choose from their own unique children which we then
        # deduplicate and merge into a closed universe

        # in the encoder direction we start with these merged deduplicated children (z's)
        # and we open them up into the open-universe choices that the encoder pi_a's want to
        # see as inputs

        # let's look at the right-side encoder bernoulli-categorical
        # the left parent of this factor crosses over to the left side pi_a
        # the right parent goes straight up to the pi_a above

        # let's look at the left-side encoder bernoulli-categorical
        # the right parent of the this factor crosses over to the right side pi_a
        # the left parent goes straight up to the pi_a above

        # all of the messages going up to pi_a's need to be categorical at this point where
        # which means that the two signals going to the left pi_a need to sum to 1
        # and the two signals going to the right pi_a also need to sum to 1

        # in terms of indexing this means
        # v[0][0] + v[1][0] = 1 # all the inputs to the left pi_a sum to 1
        # v[1][1] + v[0][1] = 1 # the inputs to the right pi_a also sum to 1

        # we already have a file for the bernoulli to categorical factor
        # what we need to figure out now is the universe factor

        # it takes in a z[:][0] Bernoulli on the left and produces two Bernoulli parents
        # u[:][0][0] bound to go straight up the left (the first index is just the heads/tails)
        # u[:][0][1] bound to cross over to the right

        # similarly the right-side universe factor takes in a Bernoulli on the right, z[:][1]  
        # and produces two Bernoulli parents:
        # u[:][1][0] bound to cross over to the left
        # u[:][1][1] bound to go straight up the right (the first index is just the heads/tails)

        # in terms of the math that the universe factor performs:
        # each one has an incoming z and two parents, left and right.

        # if parent_left=1, and child=1, then parent_right can be 1 or 0
        # as long as one parent=1 to activate the child, it doesn't matter whata the value of the other parent is
        # Writing a truth table:
        '''
        child parent_left parent_right    prob
        1     1           1               1
        1     1           0               1
        1     0           1               1
        1     0           0               0
        0     1           1               0
        0     1           0               0
        0     0           1               0
        0     0           0               1
        '''

        # z[heads/tails][below_lw]
        # u[heads/tails][below_lw][above_lw]
        # so when we are considering parent u[a][b][c] we are using child z[][b]
        
        '''
        The inputs of the universe factor are in terms of the child and the other parent
        but since the other parent, when we are forward marginalizing the encoder
        does not yet have any backward information, it is always uniformative, 
        in other words on the right hand side we always have, u[][][] = 0.5
        That will save us a lot of work.
        
        # p(parent_a = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][0][0] = z[1][0] * u[][][] + z[1][0] * u[][][]
        # p(parent_a = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][0][0] = z[][] * u[][][] + z[1][0] * u[][][]

        # p(parent_b = 1) = p(child = 1)p(parent_a = 0) + p(child = 1)p(parent_a = 1)
        u[1][0][1] = z[][] * u[][][] + z[1][0] * u[][][]
        # p(parent_b = 0) = p(child = 1)p(parent_a = 1) + p(child = 0)p(parent_a = 0)
        u[0][0][1] = z[][] * u[][][] + z[1][0] * u[][][]

        We convert all u[][][]'s on the right hand side to 0.5:

        # TODO:
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
