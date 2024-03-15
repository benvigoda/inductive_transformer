import jax
import jax.numpy as jnp  # type: ignore
import numpy as np

STRONG = 1.  # Amplify the signal
WEAK = 1e-9  # Dampen the signal


def update_weights(params):
    strong = STRONG  # Amplify the signal
    weak = WEAK  # Dampen the signal
    # Get shapes:
    num_positions, vocab_size, layer_width = params["params"]["encoders_0"]["encoder_token_pi"]["weights"].shape

    """Update weights."""
    updated_params = params
    mask_type = int
    set_weights = jax.tree_util.tree_map(lambda x: jnp.ones_like(x, dtype=mask_type), params)
    num_layer = 0
    done = False
    while not done:
        encoder_layer_key = f"encoders_{num_layer}"
        if encoder_layer_key in updated_params["params"]:
            # Get the shape of the original weights
            old_weights = updated_params["params"][encoder_layer_key]["encoder_position_pi"]["weights"]
            weights_shape = old_weights.shape  # (num_positions, layer_width)
            # Set the weights to all "weak" values
            new_weight = jnp.full(weights_shape, weak)
            new_weight = new_weight.at[num_layer].set(jnp.full(layer_width, strong))  # Match num_layer to the position in the weights
            updated_params["params"][encoder_layer_key]["encoder_position_pi"]["weights"] = new_weight
            # Note: We have constrained the model such that num_positions needs to be equal to num_layers for this setup to work right now.
            # In the future we'll have to remove that constraint

            set_weights["params"][encoder_layer_key]["encoder_position_pi"]["weights"] = jnp.zeros_like(old_weights, dtype=mask_type)

            num_layer += 1
        else:
            done = True
    lw = 0
    position = 0
    vocab_idx = 0  # should be the index for either "dog" or "cat"
    new_weight = updated_params["params"]["encoders_0"]["encoder_token_pi"]["weights"]
    new_weight = new_weight.at[position, :, lw].set(jnp.full(vocab_size, weak))
    new_weight = new_weight.at[position, vocab_idx, lw].set(strong)
    updated_params["params"]["encoders_0"]["encoder_token_pi"]["weights"] = new_weight

    return updated_params, set_weights
