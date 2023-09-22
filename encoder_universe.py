import torch  # type: ignore
from torch import nn  # type: ignore


class EncoderUniverse(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(EncoderUniverse, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

        self.u = None

    def forward(self, z):
        # z is a 2x2 tensor of Bernoulli's
        assert z.shape == (self.hyperparams.layer_width, self.hyperparams.layer_width)
        u = torch.empty(2, self.hyperparams.layer_width, self.hyperparams.layer_width)

        # we are developing an encoder closed_to_open_universe factor
        # it's output will go into an encoder bernoulli_to_categorical

        # we are going to think about what output this universe factor needs to produce (u)
        # in terms of the inputs it receives from below (z)

        # the inputs below are z's and they live in a closed universe with a limited layer width
        # the outputs go to the pi_a's above who live in an open universe of expanded choices

        # in the decoder direction, the pi_a's choose from their own unique children which we then
        # deduplicate and merge into a closed universe

        # in the encoder direction we start with these merged deduplicated children (z's)
        # and we open them up into the open-universe choices that the encoder pi_a's want to
        # see as inputs

        # let's look at the right-side encoder bernoulli-categorical
        # the left output message of this factor crosses over to the left side pi_a
        # the right output message goes straight up to the right-side i_a above

        # the left-side encoder bernoulli-categorical
        # the right output message of the this factor crosses over to the right side pi_a
        # the left output message goes straight up to the pi_a above

        # all of the messages going up to pi_a's need to be categorical at this point
        # which means that the two signals going to the left pi_a need to sum to 1
        # and the two signals going to the right pi_a also need to sum to 1

        # in terms of indexing, we would write
        # v[0][0] + v[1][0] = 1 # all the inputs to the left pi_a sum to 1
        # v[1][1] + v[0][1] = 1 # the inputs to the right pi_a also sum to 1

        # we already have a file for the bernoulli_to_categorical factor
        # what we need to figure out is the universe factor

        # in terms of the math that the universe factor performs:
        # each of the factors (left and right) has one incoming z and two parents, left and right.

        # as long as one parent=1 to activate the child, it doesn't matter what the value of the other parent is
        # for exmple, if parent_left=1, and child=1, then parent_right can be 1 or 0
        # Writing this as a truth table:
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
        # the universe factor takes in a z[:][0] Bernoulli on the left and produces two Bernoulli parents
        # u[:][0][0] bound to go straight up the left (the first index is just the heads/tails)
        # u[:][0][1] bound to cross over to the right

        # similarly the right-side universe factor takes in a Bernoulli on the right, z[:][1]
        # and produces two Bernoulli parents:
        # u[:][1][0] bound to cross over to the left
        # u[:][1][1] bound to go straight up the right (the first index is just the heads/tails)

        # u[heads/tails][below_lw][above_lw]
        # z[heads/tails][below_lw]

        # so when we are considering parent u[:][below_lw][] we are using child z[:][below_lw]

        '''
        The inputs of the universe factor are in terms of the child and the other parent
        but since the other parent, when we are forward marginalizing the encoder
        does not yet have any backward information, it is always uninformative,
        in other words on the right hand side we always have, u[][][] = 0.5
        That will save us a lot of work. e.g.

        # p(parent_left = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][0][0] = z[1][0] * u[0][][] + z[1][0] * u[1][][]
        # p(parent_left = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][0][0] = z[1][] * u[1][][] + z[0][] * u[0][][]

        We convert all u[][][]'s on the right hand side to 0.5:

        # p(parent_left = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][0][0] = z[1][0] * 0.5 + z[1][0] * 0.5
        # p(parent_left = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][0][0] = z[1][0] * 0.5 + z[0][0] * 0.5

        # we can then eliminate all the 0.5's since they factor out and then normalize out

        # p(parent_left = 1) = p(child = 1)p(parent_b = 0) + p(child = 1)p(parent_b = 1)
        u[1][1][0] = z[1][1] + z[1][1]
        # p(parent_left = 0) = p(child = 1)p(parent_b = 1) + p(child = 0)p(parent_b = 0)
        u[0][1][0]= z[0][1] + z[1][1]

        so we just need to get the child indices right
        '''
        # the universe factor takes in a z[:][0] Bernoulli on the left and produces two Bernoulli parents
        # u[:][0][0] bound to go straight up the left (the first index is just the heads/tails)
        # u[:][0][1] bound to cross over to the right

        # similarly the right-side universe factor takes in a Bernoulli on the right, z[:][1]
        # and produces two Bernoulli parents:
        # u[:][1][0] bound to cross over to the left
        # u[:][1][1] bound to go straight up the right (the first index is just the heads/tails)

        # in summary:
        # u[heads/tails][below_lw][above_lw]
        # z[heads/tails][below_lw]

        # the second index of the child is left/right
        # same index as the parent indices for left/right which is the middle index dim=1?

        # left side below
        # p(parent_left = 1) = p(child = 1)p(parent_right = 0) + p(child = 1)p(parent_right = 1)
        u[1][0][0] = z[1][0] + z[1][0]
        # p(parent_left = 0) = p(child = 1)p(parent_right = 1) + p(child = 0)p(parent_right = 0)
        u[0][0][0] = z[1][0] + z[0][0]

        # p(parent_right = 1) = p(child = 1)p(parent_left = 0) + p(child = 1)p(parent_left = 1)
        u[1][0][1] = z[1][0] + z[1][0]
        # p(parent_right = 0) = p(child = 1)p(parent_left = 1) + p(child = 0)p(parent_left = 0)
        u[0][0][1] = z[1][0] + z[0][0]

        # right side below
        # p(parent_left = 1) = p(child = 1)p(parent_right = 0) + p(child = 1)p(parent_right = 1)
        u[1][1][0] = z[1][1] + z[1][1]
        # p(parent_left = 0) = p(child = 1)p(parent_right = 1) + p(child = 0)p(parent_right = 0)
        u[0][1][0] = z[1][1] + z[0][1]

        # p(parent_right = 1) = p(child = 1)p(parent_left = 0) + p(child = 1)p(parent_left = 1)
        u[1][1][1] = z[1][1] + z[1][1]
        # p(parent_right = 0) = p(child = 1)p(parent_left = 1) + p(child = 0)p(parent_left = 0)
        u[0][1][1] = z[1][1] + z[0][1]

        u = nn.functional.normalize(u, p=1, dim=0)
        self.u = u
        return u
