import jax
import jax.numpy as jnp


def sample(key: jax.Array, decoder_t: jax.Array, temperature: float = 1.0):
    assert decoder_t.ndim == 2
    num_positions, vocab_size = decoder_t.shape

    # The categorical distribution over tokens (at each position) should sum to one.
    sums = decoder_t.sum(axis=1)
    if not jnp.allclose(sums, 1.0):
        print("WARNING: categorical distributions do not all sum to one")
        print(sums)

    # For now we just generate a single sample.
    # TODO: Allow taking multiple (independent) samples.
    log_probs = jnp.log(decoder_t) / temperature
    samples = jax.random.categorical(key, log_probs, axis=-1)
    return samples
