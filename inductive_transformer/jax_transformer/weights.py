import jax
import jax.numpy as jnp  # type: ignore

strong = 1.  # Amplify the signal
weak = 1e-9  # Dampen the signal

mask_type = int


def update_position_pi_weights(layer, params, updated_params, set_weights, prefix, layer_width):
    layer_key = f"{prefix}s_{layer}"
    position_pi = f"{prefix}_position_pi"
    if layer_key in updated_params["params"]:
        # Get the shape of the original weights
        old_weights = updated_params["params"][layer_key][position_pi]["weights"]
        weights_shape = old_weights.shape  # (num_positions, layer_width)
        # Set the weights to all "weak" values
        new_weight = jnp.full(weights_shape, weak)
        new_weight = new_weight.at[layer].set(jnp.full(layer_width, strong))  # Match num_layer to the position in the weights
        updated_params["params"][layer_key][position_pi]["weights"] = new_weight
        # Note: We have constrained the model such that num_positions needs to be equal to num_layers for this setup to work right now.
        # In the future we'll have to remove that constraint

        set_weights["params"][layer_key][position_pi]["weights"] = jnp.zeros_like(old_weights, dtype=mask_type)


def update_weights(params, vocab):
    # Get shapes:
    num_positions, vocab_size, layer_width = params["params"]["encoders_0"]["encoder_token_pi"]["weights"].shape
    assert vocab_size == len(vocab)
    """Update weights."""
    updated_params = params
    set_weights = jax.tree_util.tree_map(lambda x: jnp.ones_like(x, dtype=mask_type), params)

    num_layers = 0
    while True:
        if f"encoders_{num_layers}" not in params["params"]:
            break
        num_layers += 1

    for layer in range(num_layers):
        update_position_pi_weights(layer, params, updated_params, set_weights, "decoder", layer_width)
        update_position_pi_weights(layer, params, updated_params, set_weights, "encoder", layer_width)

    """ Update the weights for layer 1 """
    # encoders_1 is layer=1
    for lw, target_word in zip(range(layer_width), ['dog', 'cat']):
        position = 1
        vocab_idx = next((i for i, word in enumerate(vocab) if word.lower() == target_word), None)
        new_weight = updated_params["params"]["encoders_1"]["encoder_token_pi"]["weights"]
        new_weight = new_weight.at[position, :, lw].set(jnp.full(vocab_size, weak))
        new_weight = new_weight.at[position, vocab_idx, lw].set(strong)
        updated_params["params"]["encoders_1"]["encoder_token_pi"]["weights"] = new_weight

    # Fix set_weights so the gradient does not update the weights
    set_weights["params"]["encoders_1"]["encoder_token_pi"]["weights"] = jnp.zeros_like(
        updated_params["params"]["encoders_1"]["encoder_token_pi"]["weights"], dtype=mask_type
    )

    # decoder_1 is layer=1
    for lw, target_word in zip(range(layer_width), ['dog', 'cat']):
        position = 1
        vocab_idx = next((i for i, word in enumerate(vocab) if word.lower() == target_word), None)
        new_weight = updated_params["params"]["decoders_1"]["decoder_token_pi"]["weights"]
        new_weight = new_weight.at[position, :, lw].set(jnp.full(vocab_size, weak))
        new_weight = new_weight.at[position, vocab_idx, lw].set(strong)
        updated_params["params"]["decoders_1"]["decoder_token_pi"]["weights"] = new_weight
    # Fix set_weights so the gradient does not update the weights
    set_weights["params"]["decoders_1"]["decoder_token_pi"]["weights"] = jnp.zeros_like(
        updated_params["params"]["decoders_1"]["decoder_token_pi"]["weights"], dtype=mask_type
    )

    """Set the weights for layer 0"""
    # encoders_0 is layer=0
    position = 0  # layer=0 is always position=0

    lw = 0
    new_weight = updated_params["params"]["encoders_0"]["encoder_token_pi"]["weights"]
    new_weight = new_weight.at[position, :, lw].set(jnp.full(vocab_size, weak))
    vocab_idx = next((i for i, word in enumerate(vocab) if word.lower() == 'big'), None)
    new_weight = new_weight.at[position, vocab_idx, lw].set(strong)
    vocab_idx = next((i for i, word in enumerate(vocab) if word.lower() == 'large'), None)
    new_weight = new_weight.at[position, vocab_idx, lw].set(strong)
    updated_params["params"]["encoders_0"]["encoder_token_pi"]["weights"] = new_weight

    lw = 1
    vocab_idx = next((i for i, word in enumerate(vocab) if word.lower() == 'small'), None)
    new_weight = updated_params["params"]["encoders_0"]["encoder_token_pi"]["weights"]
    new_weight = new_weight.at[position, :, lw].set(jnp.full(vocab_size, weak))
    new_weight = new_weight.at[position, vocab_idx, lw].set(strong)
    updated_params["params"]["encoders_0"]["encoder_token_pi"]["weights"] = new_weight
    # Fix set_weights so the gradient does not update the weights
    set_weights["params"]["encoders_0"]["encoder_token_pi"]["weights"] = jnp.zeros_like(
        updated_params["params"]["encoders_0"]["encoder_token_pi"]["weights"], dtype=mask_type
    )

    # decoder_0 is layer=0
    lw = 0
    position = 0
    new_weight = updated_params["params"]["decoders_0"]["decoder_token_pi"]["weights"]
    new_weight = new_weight.at[position, :, lw].set(jnp.full(vocab_size, weak))

    vocab_idx = next((i for i, word in enumerate(vocab) if word.lower() == 'big'), None)
    new_weight = new_weight.at[position, vocab_idx, lw].set(strong)
    vocab_idx = next((i for i, word in enumerate(vocab) if word.lower() == 'large'), None)
    new_weight = new_weight.at[position, vocab_idx, lw].set(strong)
    updated_params["params"]["decoders_0"]["decoder_token_pi"]["weights"] = new_weight

    lw = 1
    position = 0
    new_weight = updated_params["params"]["decoders_0"]["decoder_token_pi"]["weights"]
    new_weight = new_weight.at[position, :, lw].set(jnp.full(vocab_size, weak))
    vocab_idx = next((i for i, word in enumerate(vocab) if word.lower() == 'small'), None)
    new_weight = new_weight.at[position, vocab_idx, lw].set(strong)
    updated_params["params"]["decoders_0"]["decoder_token_pi"]["weights"] = new_weight
    # Fix set_weights so the gradient does not update the weights
    set_weights["params"]["decoders_0"]["decoder_token_pi"]["weights"] = jnp.zeros_like(
        updated_params["params"]["decoders_0"]["decoder_token_pi"]["weights"], dtype=mask_type
    )

    """Set attention weights."""
    # we want no cross connections
    # for the connection between layer=1 and layer=0,
    # in lw=0, attention weight connecting to lw=0 should be strong and connecting to lw=1 weak
    # in lw=1, attention weight connecting to lw=0 should be weak and connecting to lw=1 strong
    # for the connection from layer=0 to the inputs/outputs, we can have all the weights be uniform
    new_weight = updated_params["params"]["encoders_0"]["encoder_attention_pi"]["weights"]
    new_weight = new_weight.at[:, :].set(jnp.full((layer_width, layer_width), strong / 2))
    updated_params["params"]["encoders_0"]["encoder_attention_pi"]["weights"] = new_weight
    # Fix set_weights so the gradient does not update the weights
    set_weights["params"]["encoders_0"]["encoder_attention_pi"]["weights"] = jnp.zeros_like(
        updated_params["params"]["encoders_0"]["encoder_attention_pi"]["weights"], dtype=mask_type
    )

    new_weight = updated_params["params"]["decoders_0"]["decoder_attention_pi"]["weights"]
    new_weight = new_weight.at[:, :].set(jnp.full((layer_width, layer_width), strong / 2))
    updated_params["params"]["decoders_0"]["decoder_attention_pi"]["weights"] = new_weight
    # Fix set_weights so the gradient does not update the weights
    set_weights["params"]["decoders_0"]["decoder_attention_pi"]["weights"] = jnp.zeros_like(
        updated_params["params"]["decoders_0"]["decoder_attention_pi"]["weights"], dtype=mask_type
    )

    new_weight = updated_params["params"]["encoders_1"]["encoder_attention_pi"]["weights"]
    new_weight = new_weight.at[:, :].set(jnp.full((layer_width, layer_width), weak))
    new_weight = new_weight.at[0, 0].set(strong)
    new_weight = new_weight.at[1, 1].set(strong)
    updated_params["params"]["encoders_1"]["encoder_attention_pi"]["weights"] = new_weight
    # Fix set_weights so the gradient does not update the weights
    set_weights["params"]["encoders_1"]["encoder_attention_pi"]["weights"] = jnp.zeros_like(
        updated_params["params"]["encoders_1"]["encoder_attention_pi"]["weights"], dtype=mask_type
    )

    new_weight = updated_params["params"]["decoders_1"]["decoder_attention_pi"]["weights"]
    new_weight = new_weight.at[:, :].set(jnp.full((layer_width, layer_width), weak))
    new_weight = new_weight.at[0, 0].set(strong)
    new_weight = new_weight.at[1, 1].set(strong)
    updated_params["params"]["decoders_1"]["decoder_attention_pi"]["weights"] = new_weight
    # Fix set_weights so the gradient does not update the weights
    set_weights["params"]["decoders_1"]["decoder_attention_pi"]["weights"] = jnp.zeros_like(
        updated_params["params"]["decoders_1"]["decoder_attention_pi"]["weights"], dtype=mask_type
    )

    return updated_params, set_weights
