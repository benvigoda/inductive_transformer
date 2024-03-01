from decoder_and import DecoderAnd
import jax
import numpy as np


if __name__ == "__main__":
    # Initialize RNG state.
    np_rng = np.random.default_rng()
    seed = np_rng.integers(0, 2**32 - 1)
    print(f"seed: {seed}")
    key = jax.random.PRNGKey(seed)

    bernoulli_width = 2
    layer_width = 2

    decoder_and = DecoderAnd(layer_width=layer_width)

    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (bernoulli_width, layer_width))
    y_encoder = jax.random.normal(subkey, (bernoulli_width, layer_width))
    z_encoder = jax.random.normal(subkey, (bernoulli_width, layer_width))

    print("z", z)
    print("y_encoder", y_encoder)
    print("z_encoder", z_encoder)
    x = decoder_and(z, z_encoder, y_encoder)
    print("x", x)
