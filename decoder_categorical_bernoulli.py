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
        assert v.shape == (1, self.hyperparams.layer_width, self.hyperparams.layer_width)
        u = torch.empty((2, self.hyperparams.layer_width, self.hyperparams.layer_width))
        # two parents of left open universe:
        u[1][0][0] = v[0][0]  # heads from left above
        u[0][0][0] = v[1][0]  # tails from left above
        # is the probability above that is normalized with heads from above
        # which is the below_lw=1 index

        u[1][0][1] = v[0][1]  # heads from left above
        u[0][0][1] = v[1][1]  # tails from left above

        # to parents of right open universe:
        u[1][1][0] = v[1][0]
        u[0][1][0] = v[0][0]

        u[1][1][1] = v[1][1]
        u[0][1][1] = v[0][1]

        u = nn.functional.normalize(u, p=1, dim=0)
        return u
