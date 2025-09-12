import jax.numpy as jnp
from jax.nn import logsumexp


def temperature_controlled_aggregation(decoder_t, temperature=0.1):
    """
    Use temperature to control branch competition.
    Low temperature → more like max (winner-take-all)
    High temperature → more like mean (cooperation)

    Start with high temperature and anneal to low temperature during training.
    """
    # decoder_t shape: (num_layers, num_positions, vocab_size, layer_width)

    # First aggregate across layers normally
    layer_aggregated = logsumexp(decoder_t, axis=0)

    # Apply temperature scaling for branch competition
    scaled = layer_aggregated / temperature

    # Use logsumexp with temperature (becomes more like max as T→0)
    output = logsumexp(scaled, axis=-1) * temperature

    return output


def adaptive_branch_aggregation(decoder_t, training_step, total_steps):
    """
    Gradually transition from cooperative (logsumexp) to competitive (max) aggregation.
    """
    # Calculate annealing factor (0 → 1 over training)
    progress = training_step / total_steps

    # Temperature annealing: start at 1.0, end at 0.01
    temperature = 1.0 * (1 - progress) + 0.01 * progress

    return temperature_controlled_aggregation(decoder_t, temperature)
