import jax
import jax.numpy as jnp
import numpy as np
from jax_transformer.model import BatchedInductiveTransformer


if __name__ == "__main__":
    # Initialize RNG state.
    np_rng = np.random.default_rng()
    seed = np_rng.integers(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    print(f"seed: {seed}\n")

    bernoulli_width = 2
    num_positions = 2
    vocab_size = 4
    layer_width = 2
    num_layers = 3

    key, subkey = jax.random.split(key)

    model = BatchedInductiveTransformer(
        layer_width=layer_width,
        num_positions=num_positions,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )

    key, subkey_0, subkey_1, subkey_2 = jax.random.split(key, 4)
    z_in = jax.random.uniform(
        subkey_0, minval=0.0, maxval=1.0, shape=(bernoulli_width, layer_width)
    )
    t_in = jax.random.uniform(
        subkey_1,
        minval=0.0,
        maxval=1.0,
        # The first axis specifies the batch size. Since all params are shared over the batch axis,
        # the batch size isn't consequential for initialization. (But it does matter for inference.)
        shape=(4, num_layers, num_positions, vocab_size, layer_width),
    )
    params = model.init(subkey_2, z_in, t_in)

    z_out, t_out, encoder_activations, decoder_activations = (
        model.apply(params, z_in, t_in)
    )

    print(z_out)
