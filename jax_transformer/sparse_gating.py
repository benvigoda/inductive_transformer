import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import logsumexp


class SparseGating(nn.Module):
    """
    Learn sparse gates that activate only one branch per sentence type.
    Uses Gumbel-Softmax for differentiable discrete selection.
    """
    layer_width: int
    temperature: float = 0.1

    @nn.compact
    def __call__(self, decoder_t, key):
        # decoder_t shape: (num_layers, num_positions, vocab_size, layer_width)

        # Aggregate across layers first
        layer_agg = logsumexp(decoder_t, axis=0)
        # Shape: (num_positions, vocab_size, layer_width)

        # Compute gating scores based on content
        # Use the first position as a "sentence type" indicator
        first_pos = layer_agg[0]  # (vocab_size, layer_width)

        # Learn gates based on which words are active in first position
        gate_weights = self.param(
            "gate_weights",
            nn.initializers.normal(0.01),
            (self.layer_width,)
        )

        # Add Gumbel noise for exploration during training
        if key is not None:
            gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(
                key, gate_weights.shape, minval=1e-10, maxval=1.0
            )))
            gate_scores = gate_weights + gumbel_noise
        else:
            gate_scores = gate_weights

        # Apply softmax with temperature (Gumbel-Softmax trick)
        gates = jax.nn.softmax(gate_scores / self.temperature)

        # Apply gates to branches
        gated_output = layer_agg * gates.reshape(1, 1, -1)

        # Final aggregation
        output = logsumexp(gated_output, axis=-1)

        return output, gates
