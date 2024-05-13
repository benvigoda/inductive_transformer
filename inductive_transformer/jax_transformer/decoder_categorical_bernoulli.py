from dataclasses import dataclass
import jax.numpy as jnp  # type: ignore

from inductive_transformer.jax_transformer.helper_functions import custom_normalize


@dataclass
class DecoderCategoricalBernoulli:
    layer_width: int

    def __call__(self, v):
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
        assert v.shape == (self.layer_width, self.layer_width)
        # v = torch.transpose(v, 0, 1)  # FIXME XXX FINDME THIS HELPS SOME BREAKS SOME

        # The probability of a bernoulli variable being true is the same as the probability of the
        # corresponding categorical state.
        u_1 = v

        # The probability of a bernoulli variable being false is the sum of the probabilities of all
        # the other categorical states.
        # Note: if v[i][j] is much larger than v[i][k] for k != j, then this method of performing
        # the calculation introduces a lot of rounding error.
        u_0 = jnp.sum(v, axis=-1, keepdims=True) - v

        u = jnp.stack([u_0, u_1], axis=0)

        # import pdb; pdb.set_trace()
        # u = custom_normalize(u, axis=0)  # IMPORTANT: We commented this line out because it was over correcting the weights to converge
        # to a zero loss with two sentences "big dog. small cat." Once we commented it out, it yielded a loss of 1e-10 right away, without
        # needing to train away from that point.
        assert u.shape == (2, self.layer_width, self.layer_width)
        return u
