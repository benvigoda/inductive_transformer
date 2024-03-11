from flax.training import train_state
from pprint import pprint
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pathlib

from model import BatchedInductiveTransformer
from text_parsing import InputData, ProbTensors
from weights import update_weights


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

    tx = optax.adam(learning_rate=1.0e-3)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def apply_model(state, z_in, t_in):
    """Computes gradients and loss for a single instance (not yet batched)."""

    def loss_fn(params):
        z_out, t_out = state.apply_fn(params, z_in, t_in)
        t_in_sums = jnp.sum(t_in, axis=-1)
        t_out_sums = jnp.sum(t_out, axis=-1)
        return jnp.mean(jnp.square(t_out_sums - t_in_sums))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return grads, loss


@jax.jit
def update_model(state, grads):
    # TODO We can zero out gradients here if there are parameters we don't want to update.
    return state.apply_gradients(grads=grads)


def parse_args():
    parser = argparse.ArgumentParser(description="Model arguments")
    parser.add_argument(
        "training_text", type=pathlib.Path
    )  # A text file of sentences to train on
    parser.add_argument(
        "inference_text", type=pathlib.Path
    )  # A text file of sentences to run inference on
    parser.add_argument("--layer_width", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
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
    prob_tensors = ProbTensors(data=data, layer_width=args.layer_width, print_flag=False)
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

    # Initialize all training state (most importantly, the model parameters and optimizer).
    key, subkey = jax.random.split(key)
    state = create_train_state(
        subkey,
        vocab_size=prob_tensors.vocab_size,
        num_positions=prob_tensors.num_positions,
        layer_width=args.layer_width,
        num_layers=args.num_layers,
    )
    # print(apply_model(state, prob_tensors.attention_input, all_t_tensors))

    # Train the model.
    n_training_steps = 10000
    print_every = 100
    key, train_key = jax.random.split(key)
    for step in range(0, n_training_steps):
        subkey = jax.random.fold_in(train_key, step)
        grads, loss = apply_model(state, prob_tensors.attention_input, all_t_tensors)
        state = update_model(state, grads)
        if step % print_every == 0:
            print(f"step {step}, loss: {loss:.3e}")
