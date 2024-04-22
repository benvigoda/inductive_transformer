from flax.training import train_state
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pathlib

from jax_transformer.model import BatchedInductiveTransformer
from jax_transformer.text_parsing import InputData, ProbTensors
from jax_transformer.weights import update_weights


class TrainState(train_state.TrainState):
    """A custom TrainState class that includes a `grad_mask` attribute."""

    grad_mask: jnp.ndarray


def create_train_state(key, num_positions, vocab_size, layer_width, num_layers):
    """Creates initial `TrainState`."""
    bernoulli_width = 2

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
        # The one here specifies the batch size. Since all params are shared over the batch axis,
        # the batch size isn't consequential for initialization.
        shape=(1, num_layers, num_positions, vocab_size, layer_width),
    )
    params = model.init(subkey_2, z_in, t_in)

    # Update weights.
    params, set_weights = update_weights(params)

    tx = optax.adam(learning_rate=1.0e-4)

    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, grad_mask=set_weights
    )


@jax.jit
def apply_model(state, z_in, t_in):
    """Computes gradients and loss for a single instance (not yet batched)."""

    def loss_fn(params):
        z_out, t_out, encoder_activations, decoder_activations = state.apply_fn(
            params, z_in, t_in
        )
        t_in_sums = jnp.sum(t_in, axis=-1)
        t_out_sums = jnp.sum(t_out, axis=-1)
        return jnp.mean(jnp.square(t_out_sums - t_in_sums))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return grads, loss


@jax.jit
def update_model(state, grads):
    # Zero out the gradients of parameters that we don't want to update.
    grads = jax.tree_util.tree_map(lambda x, y: x * y, grads, state.grad_mask)
    return state.apply_gradients(grads=grads)


def parse_args():
    parser = argparse.ArgumentParser(description="Model arguments")
    parser.add_argument(
        "training_text", type=pathlib.Path
    )  # A text file of sentences to train on
    parser.add_argument(
        "inference_text", type=pathlib.Path
    )  # A text file of sentences to run inference on
    parser.add_argument("--layer_width", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse args.
    args = parse_args()

    # Initialize RNG state.
    np_rng = np.random.default_rng()
    seed = np_rng.integers(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    print(f"seed: {seed}\n")

    # Load training data.
    data = InputData(args.training_text, args.inference_text, print_vals=False)
    prob_tensors = ProbTensors(
        data=data, layer_width=args.layer_width, print_flag=False
    )
    training_data = prob_tensors.format_training_data(num_layers=args.num_layers)
    # Collect all input t tensors.
    all_t_tensors = jnp.stack([example[0] for example in training_data], axis=0)
    assert all_t_tensors.shape == (
        len(training_data),
        args.num_layers,
        prob_tensors.num_positions,
        prob_tensors.vocab_size,
        args.layer_width,
    )
    print(f"vocab: {data.vocab}")

    # temp: duplicate our training data
    all_t_tensors = jnp.concatenate([all_t_tensors] * 10, axis=0)

    # Initialize all training state (most importantly, the model parameters and optimizer).
    key, subkey = jax.random.split(key)
    state = create_train_state(
        subkey,
        vocab_size=prob_tensors.vocab_size,
        num_positions=prob_tensors.num_positions,
        layer_width=args.layer_width,
        num_layers=args.num_layers,
    )

    # Train the model.
    n_epochs = 1000
    batch_size = 1
    n_steps_per_epoch = all_t_tensors.shape[0] // batch_size
    print_every = 250
    print(f"{n_epochs} epochs, {n_steps_per_epoch} steps per epoch")
    key, subkey = jax.random.split(key)
    for epoch in range(n_epochs):
        # Shuffle the data.
        shuffle_key = jax.random.fold_in(subkey, epoch)
        all_t_tensors = jax.random.permutation(shuffle_key, all_t_tensors)

        for step in range(0, n_steps_per_epoch):
            batch_data = all_t_tensors[step * batch_size : (step + 1) * batch_size]
            grads, loss = apply_model(state, prob_tensors.attention_input, batch_data)
            state = update_model(state, grads)
            if step > 0 and step % print_every == 0:
                print(f"step {step}, loss: {loss:.3e}")

        print(f"epoch {epoch}, loss: {loss:.3e}")

    # Print trained weights.
    decoder_layers = ["decoders_0", "decoders_1"]
    encoder_layers = ["encoders_0", "encoders_1"]
    decoder_sublayers = [
        "decoder_attention_pi",
        "decoder_position_pi",
        "decoder_token_pi",
    ]
    encoder_sublayers = [
        "encoder_attention_pi",
        "encoder_position_pi",
        "encoder_token_pi",
    ]

    print("===================== Decoder Layers ======================")
    for layer in decoder_layers:
        print(layer)
        layer_params = state.params["params"][layer]
        for sublayer in decoder_sublayers:
            print(sublayer)
            print(layer_params[sublayer]["weights"])
        print("")

    print("===================== Encoder Layers ======================")
    for layer in encoder_layers:
        print(layer)
        layer_params = state.params["params"][layer]
        for sublayer in encoder_sublayers:
            print(sublayer)
            print(layer_params[sublayer]["weights"])
        print("")

    # Load inference examples.
    inference_data = prob_tensors.make_inference_prompt_tensors(
        num_layers=args.num_layers
    )
    all_inference_data = jnp.stack(inference_data, axis=0)
    n_examples = len(inference_data)
    assert all_inference_data.shape == (
        n_examples,
        args.num_layers,
        prob_tensors.num_positions,
        prob_tensors.vocab_size,
        args.layer_width,
    )

    # Run inference.
    decoder_z, decoder_t, encoder_activations, decoder_activations = state.apply_fn(
        state.params, prob_tensors.attention_input, all_inference_data
    )

    print("===================== Inference Activations ======================")

    encoder_activation_keys = [
        "u",
        "v",
        "y_categorical",
        "rho_categorical",
        "x_categorical",
        "y_bernoulli",
        "x_bernoulli",
        "z_prime",
    ]
    decoder_activation_keys = [
        "x_bernoulli",
        "y_bernoulli",
        "x_categorical",
        "y_categorical",
        "v",
        "rho_categorical",
        "t_categorical",
        "u",
        "z",
    ]

    for idx in range(n_examples):
        print("--------------------------")
        print(f"Inference example {idx}")

        print("input t")
        print(all_inference_data[idx])
        print("output t")
        print(decoder_t[idx])
        print("")

        for layer_idx, layer_activation in enumerate(encoder_activations):
            print(f"Layer {layer_idx} encoder")
            for key in encoder_activation_keys:
                print(key)
                print(layer_activation[key][idx])
                print("")

        for layer_idx, layer_activation in enumerate(decoder_activations):
            print(f"Layer {layer_idx} decoder")
            for key in decoder_activation_keys:
                print(key)
                print(layer_activation[key][idx])
                print("")

    # import pdb
    # pdb.set_trace()
