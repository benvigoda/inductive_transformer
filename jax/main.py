import jax
import numpy as np

from decoder_and import DecoderAnd
from  decoder_attention_pi import DecoderAttentionPi


if __name__ == "__main__":
    # Initialize RNG state.
    np_rng = np.random.default_rng()
    seed = np_rng.integers(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    print(f"seed: {seed}\n")

    bernoulli_width = 2
    layer_width = 2

    print("Decoder And")
    decoder_and = DecoderAnd(layer_width=layer_width)
    key, subkey_0, subkey_1, subkey_2 = jax.random.split(key, 4)
    z = jax.random.normal(subkey_0, (bernoulli_width, layer_width))
    y_encoder = jax.random.normal(subkey_1, (bernoulli_width, layer_width))
    z_encoder = jax.random.normal(subkey_2, (bernoulli_width, layer_width))
    print("z", z)
    print("y_encoder", y_encoder)
    print("z_encoder", z_encoder)
    x, y = decoder_and(z, z_encoder, y_encoder)
    print("x", x)
    print("y", y)
    print("")

    print("Decoder Attention Pi")
    decoder_attention = DecoderAttentionPi(layer_width=layer_width)
    y = jax.random.normal(subkey_0, (1, layer_width))
    params = decoder_attention.init(subkey_0, y)
    v = decoder_attention.apply(params, y)
    print("params", params["params"])
    print("y", y)
    print("v", v)
    print("")
