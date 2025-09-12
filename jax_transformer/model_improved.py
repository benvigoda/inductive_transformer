# Simple modification to model.py aggregation
import jax.numpy as jnp
from jax.nn import logsumexp


def improved_aggregation(decoder_t, method='sharp_softmax'):
    """
    Different aggregation methods to reduce branch mixing.
    """
    # decoder_t shape: (num_layers, num_positions, vocab_size, layer_width)

    # First aggregate across layers (they should cooperate)
    layer_agg = logsumexp(decoder_t, axis=0)

    if method == 'sharp_softmax':
        # Use low temperature to sharpen selection
        temperature = 0.05
        scaled = layer_agg / temperature
        output = logsumexp(scaled, axis=-1) * temperature

    elif method == 'weighted_max':
        # Mix between max and mean
        alpha = 0.9  # How much to weight toward max
        max_val = jnp.max(layer_agg, axis=-1)
        mean_val = logsumexp(layer_agg, axis=-1)
        output = alpha * max_val + (1 - alpha) * mean_val

    elif method == 'top_k':
        # Only use top branch
        k = 1
        top_k_indices = jnp.argsort(layer_agg, axis=-1)[..., -k:]
        mask = jnp.zeros_like(layer_agg)
        mask = mask.at[..., top_k_indices].set(1.0)
        masked = jnp.where(mask, layer_agg, -jnp.inf)
        output = logsumexp(masked, axis=-1)

    else:  # standard
        output = logsumexp(layer_agg, axis=-1)

    return output
