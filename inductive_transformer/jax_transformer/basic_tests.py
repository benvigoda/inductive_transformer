from pprint import pprint
import jax
import numpy as np

from inductive_transformer.jax_transformer.decoder_and import DecoderAnd
from inductive_transformer.jax_transformer.decoder_attention_pi import DecoderAttentionPi
from inductive_transformer.jax_transformer.decoder_bernoulli_categorical import DecoderBernoulliCategorical
from inductive_transformer.jax_transformer.decoder_categorical_bernoulli import DecoderCategoricalBernoulli
from inductive_transformer.jax_transformer.decoder_position_pi import DecoderPositionPi
from inductive_transformer.jax_transformer.decoder_token_pi import DecoderTokenPi
from inductive_transformer.jax_transformer.decoder_universe import DecoderUniverse
from inductive_transformer.jax_transformer.decoder_layer import DecoderLayer

from inductive_transformer.jax_transformer.encoder_and import EncoderAnd
from inductive_transformer.jax_transformer.encoder_attention_pi import EncoderAttentionPi
from inductive_transformer.jax_transformer.encoder_bernoulli_categorical import EncoderBernoulliCategorical
from inductive_transformer.jax_transformer.encoder_categorical_bernoulli import EncoderCategoricalBernoulli
from inductive_transformer.jax_transformer.encoder_position_pi import EncoderPositionPi
from inductive_transformer.jax_transformer.encoder_token_pi import EncoderTokenPi
from inductive_transformer.jax_transformer.encoder_universe import EncoderUniverse
from inductive_transformer.jax_transformer.encoder_layer import EncoderLayer

from model import InductiveTransformer
from inductive_transformer.jax_transformer.weights_width_2_layers_2 import init_weights


def main():
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
    decoder_bernoulli_categorical = DecoderBernoulliCategorical(layer_width=layer_width)
    bernoulli = jax.random.normal(subkey, (bernoulli_width, layer_width))
    categorical = decoder_bernoulli_categorical(bernoulli)
    print("bernoulli", bernoulli)
    print("categorical", categorical)
    print("")

    print("Decoder Categorical Bernoulli")
    key, subkey = jax.random.split(key)
    decoder_categorical_bernoulli = DecoderCategoricalBernoulli(layer_width=layer_width)
    v = jax.random.normal(subkey, (layer_width, layer_width))
    u = decoder_categorical_bernoulli(v)
    print("v", v)
    print("u", u)
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

    print("Decoder Universe")
    key, subkey = jax.random.split(key)
    decoder_universe = DecoderUniverse(layer_width=layer_width)
    u = jax.random.normal(subkey, (2, layer_width, layer_width))
    z = decoder_universe(u)
    print("u", u)
    print("z", z)
    print("")

    print("Decoder Layer")
    key, subkey_0, subkey_1, subkey_2, subkey_3 = jax.random.split(key, 5)
    decoder_layer = DecoderLayer(
        layer_width=layer_width,
        num_positions=num_positions,
        vocab_size=vocab_size,
    )
    z_prime = jax.random.normal(subkey_0, (bernoulli_width, layer_width))
    x_encoder = jax.random.normal(subkey_1, (bernoulli_width, layer_width))
    y_encoder = jax.random.normal(subkey_2, (bernoulli_width, layer_width))
    params = decoder_layer.init(subkey_3, z_prime, x_encoder, y_encoder)
    z, t, activations = decoder_layer.apply(params, z_prime, x_encoder, y_encoder)
    print("params", params["params"])
    print("z_prime", z_prime)
    print("x_encoder", x_encoder)
    print("y_encoder", y_encoder)
    print("z", z)
    print("t", t)
    print("")

    print("Encoder And")
    key, subkey_0, subkey_1, subkey_2 = jax.random.split(key, 4)
    encoder_and = EncoderAnd()
    x = jax.random.normal(subkey_0, (bernoulli_width, layer_width))
    y = jax.random.normal(subkey_1, (bernoulli_width, layer_width))
    z = encoder_and(x, y)
    print("x", x)
    print("y", y)
    print("z", z)
    print("")

    print("Encoder Attention Pi")
    key, subkey_0, subkey_1 = jax.random.split(key, 3)
    encoder_attention = EncoderAttentionPi(
        layer_width=layer_width, vocab_size=vocab_size
    )
    v = jax.random.normal(subkey_0, (layer_width, layer_width))
    params = encoder_attention.init(subkey_1, v)
    y = encoder_attention.apply(params, v)
    print("params", params["params"])
    print("v", v)
    print("y", y)
    print("")

    print("Encoder Bernoulli Categorical")
    key, subkey = jax.random.split(key)
    encoder_bernoulli_categorical = EncoderBernoulliCategorical()
    u = jax.random.normal(subkey, (bernoulli_width, layer_width))
    v = encoder_bernoulli_categorical(u)
    print("u", u)
    print("v", v)
    print("")

    print("Encoder Categorical Bernoulli")
    key, subkey = jax.random.split(key)
    encoder_categorical_bernoulli = EncoderCategoricalBernoulli(layer_width=layer_width)
    v = jax.random.normal(subkey, (1, layer_width))
    u = encoder_categorical_bernoulli(v)
    print("v", v)
    print("u", u)
    print("")

    print("Encoder Position Pi")
    key, subkey_0, subkey_1 = jax.random.split(key, 3)
    encoder_position_pi = EncoderPositionPi(
        num_positions=num_positions, layer_width=layer_width
    )
    rho = jax.random.normal(subkey_0, (num_positions, layer_width))
    params = encoder_position_pi.init(subkey_1, rho)
    x = encoder_position_pi.apply(params, rho)
    print("params", params["params"])
    print("rho", rho)
    print("x", x)
    print("")

    print("Encoder Token Pi")
    key, subkey_0, subkey_1 = jax.random.split(key, 3)
    encoder_token_pi = EncoderTokenPi(
        num_positions=num_positions, layer_width=layer_width, vocab_size=vocab_size
    )
    t = jax.random.normal(subkey_0, (num_positions, vocab_size, layer_width))
    params = encoder_token_pi.init(subkey_1, t)
    rho = encoder_token_pi.apply(params, t)
    print("params", params["params"])
    print("t", t)
    print("rho", rho)
    print("")

    print("Encoder Universe")
    key, subkey = jax.random.split(key)
    encoder_universe = EncoderUniverse(layer_width=layer_width)
    z = jax.random.normal(subkey, (2, layer_width))
    u = encoder_universe(z)
    print("z", z)
    print("u", u)
    print("")

    print("Encoder Layer")
    key, subkey_0, subkey_1, subkey_2 = jax.random.split(key, 4)
    encoder_layer = EncoderLayer(
        layer_width=layer_width, num_positions=num_positions, vocab_size=vocab_size
    )
    z = jax.random.normal(subkey_0, (bernoulli_width, layer_width))
    t = jax.random.normal(subkey_1, (num_positions, vocab_size, layer_width))
    params = encoder_layer.init(subkey_2, z, t)
    z_prime, x_bernoulli, y_bernoulli, activations = encoder_layer.apply(params, z, t)
    print("params", params["params"])
    print("z", z)
    print("t", t)
    print("z_prime", z_prime)
    print("x_bernoulli", x_bernoulli)
    print("y_bernoulli", y_bernoulli)
    print("")

    print("Inductive Transformer")
    key, subkey_0, subkey_1, subkey_2 = jax.random.split(key, 4)
    inductive_transformer = InductiveTransformer(
        layer_width=layer_width,
        num_positions=num_positions,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )
    z_in = jax.random.normal(subkey_0, (bernoulli_width, layer_width))
    t_in = jax.random.normal(
        subkey_1, (num_layers, num_positions, vocab_size, layer_width)
    )
    params = inductive_transformer.init(subkey_2, z_in, t_in)
    z_out, t_out, encoder_activations, decoder_activations = (
        inductive_transformer.apply(params, z_in, t_in)
    )
    param_shapes = jax.tree_map(lambda x: x.shape, params)
    print("params (shapes)")
    pprint(param_shapes["params"])
    print("z_in", z_in)
    print("t_in", t_in)
    print("z_out", z_out)
    print("t_out", t_out)

    print("Inductive Transformer Weights Set")
    key, subkey_0, subkey_1, subkey_2 = jax.random.split(key, 4)
    inductive_transformer = InductiveTransformer(
        layer_width=layer_width,
        num_positions=num_positions,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )
    z_in = jax.random.normal(subkey_0, (bernoulli_width, layer_width))
    t_in = jax.random.normal(
        subkey_1, (num_layers, num_positions, vocab_size, layer_width)
    )
    params = inductive_transformer.init(subkey_2, z_in, t_in)
    updated_params, set_weights = set_weights(params)
    z_out, t_out, encoder_activations, decoder_activations = (
        inductive_transformer.apply(updated_params, z_in, t_in)
    )
    param_shapes = jax.tree_map(lambda x: x.shape, updated_params)
    print("params (shapes)")
    pprint(param_shapes["params"])
    print("z_in", z_in)
    print("t_in", t_in)
    print("z_out", z_out)
    print("t_out", t_out)


if __name__ == "__main__":
    main()
