import torch  # type: ignore
from torch import nn  # type: ignore
from helper_functions import custom_normalize


class DecoderCategoricalBernoulli(nn.Module):

    def __init__(self, hyperparams, active_layer: int):
        super(DecoderCategoricalBernoulli, self).__init__()
        self.hyperparams = hyperparams
        self.active_layer = active_layer

        self.u = None

    def forward(self, v):
        # v[below_lw][above_lw]
        # u[heads/tails][below_lw][above_lw]

        # there are four signals coming down.
        # the two going to the left below are:
        # v[0][0]
        # v[0][1]
        # --> u[:][0][0]

        # and two going to the right below:
        # v[1][0]
        # v[1][1]
        #  --> u[:][1][]

        # we need to convert all of these to bernoullis
        # left:

        '''
        v is size (layer_width, layer_width)
        # Note that "above" is incoming in the decoder while "below" is outgoing
        v, dim = 1 indexes across the layer above, with index above_lw which is short for "above layer width"
        v, dim = 0 indexes the choice that a given attention pi is making
        we should think of this dim=0 above choice as choosing one of the concepts in the layer below
        therefore dim=0 indexes the layer below.  we'll call that index, below_lw "below layer width"
        below_lw = above_choice

        let's think in terms of below_lw
        at below_lw=0, we receive a signal from attention_pi's at above_lw = 0 and above_lw = 1
        Those are the two parents of the decoder_univers.  But before it vcan consume them,
        we must convert both of them to Bernoullis.

        So what we need to do is to convert above_lw=0 to a Bernoulli
        The question is, what is the value of the above_choice we should grab from the pi above?
        The answer is, for below_lw=0, we want above_choice=0

        To convert categorical values to Bernoulli we take the categorical value and
        that is p(1) for the Bernoulli

        the v indexing is [below_lw][above_lw]
        the u indexing is [heads/tails][below_lw][above_lw]
        '''
        if self.active_layer == 1 and False:  # Used for testing only
            v[0][0] = 1
            v[0][1] = 1  # --> z[0]=1

            v[1][0] = 0
            v[1][1] = 0  # --> z[1]=0
            # we should observe z[0] = OR(v[0][0], v[0][1]) = 1

        assert v.shape == (self.hyperparams.layer_width, self.hyperparams.layer_width)
        # v = torch.transpose(v, 0, 1)  # FIXME XXX FINDME THIS HELPS SOME BREAKS SOME
        u = torch.empty((2, self.hyperparams.layer_width, self.hyperparams.layer_width), device=v.device)

        # The probability of a bernoulli variable being true is the same as the probability of the
        # corresponding categorical state.
        u[1] = v

        # The probability of a bernoulli variable being false is the sum of the probabilities of all
        # the other categorical states.
        # Note: if v[i][j] is much larger than v[i][k] for k != j, then this method of performing
        # the calculation introduces a lot of rounding error.
        u[0] = v.sum(dim=-1, keepdim=True) - v

        # import pdb; pdb.set_trace()
        # u = nn.functional.normalize(u, p=1, dim=0)
        u = custom_normalize(u, dim=0)

        self.u = u
        return u
