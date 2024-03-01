import jax
import numpy as np

from decoder_and import DecoderAnd
from decoder_attention_pi import DecoderAttentionPi
from decoder_bernoulli_categorical import DecoderBernoulliCategorical
from decoder_position_pi import DecoderPositionPi
from decoder_token_pi import DecoderTokenPi


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

    print("Decoder And")
    key, subkey_0, subkey_1, subkey_2 = jax.random.split(key, 4)
    decoder_and = DecoderAnd(layer_width=layer_width)
    z = jax.random.normal(subkey_0, (bernoulli_width, layer_width))
    y_encoder = jax.random.normal(subkey_1, (bernoulli_width, layer_width))
    z_encoder = jax.random.normal(subkey_2, (bernoulli_width, layer_width))
    x, y = decoder_and(z, z_encoder, y_encoder)
    print("z", z)
    print("y_encoder", y_encoder)
    print("z_encoder", z_encoder)
    print("x", x)
    print("y", y)
    print("")

    print("Decoder Attention Pi")
    key, subkey_0, subkey_1 = jax.random.split(key, 3)
    decoder_attention = DecoderAttentionPi(layer_width=layer_width)
    y = jax.random.normal(subkey_0, (1, layer_width))
    params = decoder_attention.init(subkey_1, y)
    v = decoder_attention.apply(params, y)
    print("params", params["params"])
    print("y", y)
    print("v", v)
    print("")

    print("Decoder Bernoulli Categorical")
    key, subkey = jax.random.split(key)
    decoder_categorical_bernoulli = DecoderBernoulliCategorical(layer_width=layer_width)
    bernoulli = jax.random.normal(subkey, (bernoulli_width, layer_width))
    categorical = decoder_categorical_bernoulli(bernoulli)
    print("bernoulli", bernoulli)
    print("categorical", categorical)
    print("")

    print("Decoder Position Pi")
    key, subkey_0, subkey_1 = jax.random.split(key, 3)
    decoder_position_pi = DecoderPositionPi(
        num_positions=num_positions, layer_width=layer_width
    )
    x = jax.random.normal(subkey_0, (1, layer_width))
    params = decoder_position_pi.init(subkey_1, x)
    rho = decoder_position_pi.apply(params, x)
    print("params", params["params"])
    print("x", x)
    print("rho", rho)
    print("")

    print("Decoder Token Pi")
    key, subkey_0, subkey_1 = jax.random.split(key, 3)
    decoder_token_pi = DecoderTokenPi(
        num_positions=num_positions, vocab_size=vocab_size, layer_width=layer_width
    )
    rho = jax.random.normal(subkey_0, (num_positions, layer_width))
    params = decoder_token_pi.init(subkey_1, rho)
    t = decoder_token_pi.apply(params, rho)
    print("params", params["params"])
    print("rho", rho)
    print("t", t)
    print("")
