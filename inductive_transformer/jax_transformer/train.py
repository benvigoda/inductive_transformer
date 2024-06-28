from flax.training import train_state
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
import pathlib

from inductive_transformer.jax_transformer.model import BatchedInductiveTransformer
from inductive_transformer.jax_transformer.text_parsing import InputData, ProbTensors
from inductive_transformer.jax_transformer.weights import update_weights
from inductive_transformer.jax_transformer.printing import print_params


class TrainState(train_state.TrainState):
    """A custom TrainState class that includes a `grad_mask` attribute."""

    grad_mask: jnp.ndarray


def create_train_state(
    key, num_positions, vocab, layer_width, num_layers, noise_seed=None, perturb_flag=False
):
    """Creates initial `TrainState`."""
    bernoulli_width = 2
    vocab_size = len(vocab)
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
    # If perturb_flag is True, we will set weights as defined in weights.py.
    params, set_weights = update_weights(params, vocab, set_all_weights=perturb_flag)

    key, subkey = jax.random.split(key)
    if noise_seed is None:
        tx = optax.chain(
            optax.adam(learning_rate=1.0e-4),
        )
    else:
        tx = optax.chain(
            optax.add_noise(eta=1.0e-2, gamma=0.999, seed=noise_seed),
        )
    # If perturb_flag is True, we will not update the weights set in weights.py.
    if perturb_flag:
        grad_mask = set_weights
    else:
        grad_mask = None
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        grad_mask=grad_mask,
    )
    return state, model


@jax.jit
def apply_model(state, z_in, t_in, truths):
    """Computes gradients and loss for a single instance (not yet batched)."""

    def loss_fn(params):
        z_out, t_out, encoder_activations, decoder_activations = state.apply_fn(
            params, z_in, t_in
        )
        assert t_out.shape == truths.shape
        loss = jnp.mean(jnp.square(t_out - truths))
        # jax.debug.print("t_out\n{}", t_out)
        # jax.debug.print("truths\n{}", truths)
        # jax.debug.print("loss {}\n", loss)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    return grads, loss


@jax.jit
def update_model(state, grads):
    # Zero out the gradients of parameters that we don't want to update.
    if state.grad_mask is not None:
        grads = jax.tree_util.tree_map(lambda x, y: x * y, grads, state.grad_mask)
    return state.apply_gradients(grads=grads)


def parse_args():
    parser = argparse.ArgumentParser(description="Model arguments")
    parser.add_argument(
        "training_text", type=pathlib.Path
    )  # A text file of sentences to train on
    parser.add_argument(
        "--prompt_text", type=pathlib.Path
    )  # A text file of sentences to run inference on
    # if doing perturbation test, you also need to give it training_text
    parser.add_argument("--perturb", action="store_true")

    '''
    if --train_text not empty
    train with entirely free weights

    if --prompt_text not empty
    do inference on the prompt_text
    if the prompt_text does not completely fill the context window, set the
    z_prime appropriate activations in the encoder to all 1's

    if both training_text and prompt_text are non-empty
    first train the weights with training_text,
    then run inference with the prompt text

    if both --perturb is True and training_text is non-empty:
    train but do not modify the weights that are set in weights.py

    if --perturb is True and training_text is non-empty and prompt_text is non-empty:
    train but do not modify the weights that are set in weights.py
    then run inference with the prompt_text
    if the prompt_text does not completely fill the context window, set the
    z_prime appropriate activations in the encoder to all 1's
    '''

    parser.add_argument("--layer_width", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse args.
    args = parse_args()
    # If doing perturbation test, you also need to give it training_text
    if args.perturb:
        assert args.training_text

    # Initialize RNG state.
    np_rng = np.random.default_rng()
    seed = np_rng.integers(0, 2**32 - 1)
    # seed = 11675966
    # seed = 615523631
    key = jax.random.PRNGKey(seed)
    print(f"seed: {seed}\n")

    key, subkey = jax.random.split(key)
    noise_seed = jax.random.randint(
        subkey, (1,), jnp.iinfo(jnp.int32).min, jnp.iinfo(jnp.int32).max
    )[0]
    noise_seed = None  # type: ignore  # To not include noise in the training process.

    # Load training data.
    data = InputData(args.training_text, args.prompt_text, print_vals=False)
    prob_tensors = ProbTensors(
        data=data, layer_width=args.layer_width, print_flag=False
    )
    training_data = prob_tensors.format_training_data()
    # Collect all input t tensors.
    all_t_tensors = jnp.stack([example[0] for example in training_data], axis=0)
    all_outputs = jnp.stack([example[1] for example in training_data], axis=0)
    assert all_t_tensors.shape == (
        len(training_data),
        args.num_layers,
        prob_tensors.num_positions,
        prob_tensors.vocab_size,
        args.layer_width,
    )
    assert all_outputs.shape == (
        len(training_data),
        prob_tensors.num_positions,
        prob_tensors.vocab_size,
    )
    print(f"vocab: {data.vocab}")
    print(f"num training examples: {all_t_tensors.shape[0]}")

    # Initialize all training state (most importantly, the model parameters and optimizer).
    key, subkey = jax.random.split(key)
    state, model = create_train_state(
        subkey,
        vocab=data.vocab,
        num_positions=prob_tensors.num_positions,
        layer_width=args.layer_width,
        num_layers=args.num_layers,
        noise_seed=noise_seed,
        perturb_flag=args.perturb,
    )

    # Check the initial loss.
    grads, loss = apply_model(
        state, prob_tensors.attention_input, all_t_tensors, all_outputs
    )
    print(f"initial loss: {loss:.3e}")

    # temp: duplicate our training data
    all_t_tensors = jnp.concatenate([all_t_tensors] * 100, axis=0)
    all_outputs = jnp.concatenate([all_outputs] * 100, axis=0)
    print(f"num training examples (padded): {all_t_tensors.shape[0]}")

    # Train the model.
    if args.training_text:
        n_epochs = 0
        batch_size = 10
        n_steps_per_epoch = all_t_tensors.shape[0] // batch_size
        print_every = 100
        print(f"{n_epochs} epochs, {n_steps_per_epoch} steps per epoch")
        key, subkey = jax.random.split(key)
    else:
        n_epochs = 0
    for epoch in range(n_epochs):
        # Shuffle the data.
        # shuffle_key = jax.random.fold_in(subkey, epoch)
        # all_t_tensors = jax.random.permutation(shuffle_key, all_t_tensors)
        # all_outputs = jax.random.permutation(shuffle_key, all_outputs)

        for step_idx in range(0, n_steps_per_epoch):
            start = step_idx * batch_size
            batch_input_data = all_t_tensors[start:start + batch_size]
            batch_output_data = all_outputs[step_idx * batch_size: (step_idx + 1) * batch_size]
            grads, loss = apply_model(
                state, prob_tensors.attention_input, batch_input_data, batch_output_data
            )
            state = update_model(state, grads)

        if epoch % print_every == 0:
            print(f"epoch {epoch}, loss: {loss:.3e}")

    # Print trained weights.
    print_params(state, data.vocab)

    if not args.prompt_text:
        print("No prompt text given, exiting.")
        exit()

    # Load inference examples.
    inference_data = prob_tensors.make_inference_prompt_tensors()
    all_inference_data = jnp.stack(inference_data, axis=0)
    n_examples = len(inference_data)
    assert all_inference_data.shape == (
        n_examples,
        args.num_layers,
        prob_tensors.num_positions,
        prob_tensors.vocab_size,
        args.layer_width,
    )

    # uniform distribution
    # prompt_data = all_inference_data.at[:, :, 1, :, :].set(1.0 / prob_tensors.vocab_size)
    # all words epsilon
    # prompt_data = all_inference_data.at[:, :, 1, :, :].set(1e-6)
    prompt_data = all_inference_data
    print("prompt data", prompt_data.shape)
    print(prompt_data)
    print("attention input")
    print(prob_tensors.attention_input)

    # Run inference.
    decoder_z, decoder_t, encoder_activations, decoder_activations = state.apply_fn(
        state.params, prob_tensors.attention_input, prompt_data
    )

    print("===================== Inference Activations ======================")

    encoder_activation_keys = [
        "z",
        "u",
        "v",
        "y_categorical",
        "y_bernoulli",
        "rho_categorical",
        "x_categorical",
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
        print(prompt_data[idx])
        print("output t")
        print(decoder_t[idx])
        print("")

        for layer_idx, layer_activation in enumerate(encoder_activations):
            print(f"Layer {layer_idx} encoder")
            for key in encoder_activation_keys:  # type: ignore
                print(key)
                print(layer_activation[key][idx])
                print("")

        for layer_idx, layer_activation in enumerate(decoder_activations):
            print(f"Layer {layer_idx} decoder")
            for key in decoder_activation_keys:  # type: ignore
                print(key)
                print(layer_activation[key][idx])
                print("")
