import jax
from typing import List, Optional
import jax.numpy as jnp  # type: ignore
from inductive_transformer.jax_transformer.helper_functions import EPSILON


# FIXME: decided not to implement perturbation experiments for 6-layer model yet

strong = 1.0 - EPSILON  # Amplify the signal
weak = EPSILON  # Dampen the signal

mask_type = int

# Make a class of weight parameters
class WeightParams:
    def __init__(self, params, mask):
        self.params = params
        self.mask = mask
        self.num_positions, self.vocab_size, self.layer_width = params["params"]["encoders_0"]["encoder_token_pi"]["weights"].shape



def set_position_pi_weights(
    layer: int,
    params,
    mask,
    prefix,
    layer_width,
    layers: Optional[List[int]] = None,
    columns: Optional[List[int]] = None,
    words: Optional[List[int]] = None,
):

    layer_key = f"{prefix}s_{layer}"
    position_pi = f"{prefix}_position_pi"
    if layer_key in params["params"]:
        # Get the shape of the original weights
        old_weights = params["params"][layer_key][position_pi]["weights"]
        # Set the weights to all "weak" values
        new_weights = jnp.full(old_weights.shape, weak)  # (num_positions, layer_width)
        num_positions = old_weights.shape[0]

        # Note: We have constrained the model such that num_positions needs to be equal to num_layers for this setup to work right now.
        # In the future we'll have to remove that constraint
        assert num_positions == layer_width
        assert old_weights.shape == (num_positions, layer_width)

        # Use negative layer indexing to match num_layer to the mirror image positions in the weights
        new_weights = new_weights.at[-layer - 1].set(jnp.full(layer_width, strong))
        params["params"][layer_key][position_pi]["weights"] = new_weights


        # where the mask is zero, the weight cannot be changed
        mask["params"][layer_key][position_pi]["weights"] = jnp.zeros_like(old_weights, dtype=mask_type)

        # free the position_pi weights in certain layers to train
        if layers is not None:
            mask["params"][layer_key][position_pi]["weights"] = free_positions_by_layer(layers, shape=old_weights.shape)

        # free the position_pi weights in particular column to train
        if columns is not None:
            mask["params"][layer_key][position_pi]["weights"] = free_positions_by_column(columns, shape=old_weights.shape)

        # free the position_pi weights for a particular word to train
        if words is not None:
            mask["params"][layer_key][position_pi]["weights"] = free_positions_by_word(words, shape=old_weights.shape)

    else:
        raise ValueError(f"Layer {layer_key} not found in params.")


def free_positions_by_layer(layers, shape):
    mask = jnp.ones_like(shape, dtype=mask_type)
    mask = mask.at[:, layers].set(1)
    return mask


def free_positions_by_column(columns, shape):
    raise NotImplementedError("Not implemented yet")
    

def free_positions_by_word(words, shape):
    raise NotImplementedError("Not implemented yet")


def lock_all_weights(params, vocab):
    """ Update the weights for layer 1 """
    for lw, target_words in zip(range(layer_width), [['small'], ['big']]):  # [['big', 'large'], ['small']]
        '''
        in the position where we want to listen for a particular word
        set a single word weight to strong and all the others to weak
        '''
        position = 0

        # encoders_1 is layer=1
        new_weight = updated_params["params"]["encoders_1"]["encoder_token_pi"]["weights"]
        new_weight = new_weight.at[position, :, lw].set(jnp.full(vocab_size, weak))
        for target_word in target_words:
            vocab_idx = next((i for i, word in enumerate(vocab) if word.lower() == target_word), None)
            new_weight = new_weight.at[position, vocab_idx, lw].set(strong)
        updated_params["params"]["encoders_1"]["encoder_token_pi"]["weights"] = new_weight

        # decoder_1 is layer=1
        new_weight = updated_params["params"]["decoders_1"]["decoder_token_pi"]["weights"]
        new_weight = new_weight.at[position, :, lw].set(jnp.full(vocab_size, weak))
        for target_word in target_words:
            vocab_idx = next((i for i, word in enumerate(vocab) if word.lower() == target_word), None)
            new_weight = new_weight.at[position, vocab_idx, lw].set(strong)
        updated_params["params"]["decoders_1"]["decoder_token_pi"]["weights"] = new_weight

        '''
        in the position where we want to listen for NO word
        set all weights to weak
        '''
        position = 1

        # encoders_1 is layer=1
        new_weight = updated_params["params"]["encoders_1"]["encoder_token_pi"]["weights"]
        new_weight = new_weight.at[position, :, lw].set(jnp.full(vocab_size, weak))
        updated_params["params"]["encoders_1"]["encoder_token_pi"]["weights"] = new_weight

        # decoder_1 is layer=1
        new_weight = updated_params["params"]["decoders_1"]["decoder_token_pi"]["weights"]
        new_weight = new_weight.at[position, :, lw].set(jnp.full(vocab_size, weak))
        updated_params["params"]["decoders_1"]["decoder_token_pi"]["weights"] = new_weight

        # Fix set_weights so the gradient does not update the weights
        # set_weights["params"]["encoders_1"]["encoder_token_pi"]["weights"] = jnp.zeros_like(
        #     updated_params["params"]["encoders_1"]["encoder_token_pi"]["weights"], dtype=mask_type
        # )
        # set_weights["params"]["decoders_1"]["decoder_token_pi"]["weights"] = jnp.zeros_like(
        #     updated_params["params"]["decoders_1"]["decoder_token_pi"]["weights"], dtype=mask_type
        # )


def set_weights(params, vocab, lock_all_weights=False):

    if lock_all_weights:
        lock_all_weights(params, vocab)

    set_weights_encoder(params, vocab, lock_all_weights)

    set_weights_decoder(params, vocab, lock_all_weights)


