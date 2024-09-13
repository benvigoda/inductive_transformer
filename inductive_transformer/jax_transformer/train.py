from flax.training import train_state  # type: ignore
from jax.tree_util import tree_flatten  # type: ignore
import argparse
import jax  # type: ignore
import os
import jax.numpy as jnp  # type: ignore
import numpy as np  # type: ignore
import optax  # type: ignore
import pathlib


from inductive_transformer.datasets.anavan import make_cat_dog_anavan, make_cat_dog_worm_bird_anavan  # type: ignore
from inductive_transformer.jax_transformer.model import BatchedInductiveTransformer
from inductive_transformer.jax_transformer.text_parsing import InputData, ProbTensors
from inductive_transformer.jax_transformer.weights_broad_init import init_weights
from inductive_transformer.jax_transformer.printing import (
    print_params,
    print_activations,
)
from inductive_transformer.jax_transformer.sampling import sample
from inductive_transformer.jax_transformer.histogram_generations import (
    histogram_results,
)


class TrainState(train_state.TrainState):
    """A custom TrainState class that includes a `grad_mask` attribute."""

    grad_mask: jnp.ndarray


def create_train_state(
    key,
    num_positions,
    vocab,
    layer_width,
    num_layers,
    noise_seed=None,
    initialize_weights=False,
    perturb_flag=False,
    perturb_position=None,
    perturb_token=None,
    perturb_attention=None,
    surgical_perturb=False,
    lock_all_weights=False,
    noise_value=0.01,
    zero_out_right_weights=False,
    zero_out_left_weights=False,
    catsanddogs=False,
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
    # If initialize_weights is True, we will set weights as defined in weights.py.
    grad_mask = None
    if initialize_weights:
        params, weight_mask = init_weights(
            params,
            vocab,
            lock_all_weights=lock_all_weights,
            perturb_weights=perturb_flag,
            perturb_position=perturb_position,
            perturb_token=perturb_token,
            perturb_attention=perturb_attention,
            surgical_perturb=surgical_perturb,
            noise_value=noise_value,
            zero_out_right_weights=zero_out_right_weights,
            zero_out_left_weights=zero_out_left_weights,
            catsanddogs=catsanddogs,
        )
        grad_mask = weight_mask

    key, subkey = jax.random.split(key)
    if noise_seed is None:
        # Pick a random number between 1e-8 and 1e-1
        # lr = 10 ** np.random.uniform(-8, -1)
        lr = 1e-3
        tx = optax.chain(
            optax.adam(learning_rate=lr),
        )
        # b1 means the exponential decay rate for the first moment estimates
        # b2 means the exponential decay rate for the second moment estimates
        # A lower b1 will make the optimizer more aggressive, a higher b1 will make it less aggressive
        # A lower b2 will make the optimizer more aggressive, a higher b2 will make it less aggressive
    else:
        tx = optax.chain(
            optax.add_noise(eta=1.0e-2, gamma=0.999, seed=noise_seed),
        )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        grad_mask=grad_mask,
    )
    num_params = count_params(params)
    print(f"Number of parameters: {num_params}")
    return state, model, lr


@jax.jit
def apply_model(state, z_in, t_in, truths):
    """Computes gradients and loss for a single instance (not yet batched)."""

    def loss_fn(params):
        z_out, t_out, encoder_activations, decoder_activations = state.apply_fn(
            params, z_in, t_in
        )
        assert t_out.shape == truths.shape
        # loss = jnp.mean(jnp.square(t_out - truths))
        # Use cross entropy loss
        import optax
        loss = optax.softmax_cross_entropy(jnp.log(t_out), truths).mean()
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


def run_and_print_inference(
    state,
    prob_tensors,
    args,
    activations_file_name,
    folder_name=None,
    silence_print=False,
):
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
    if not silence_print:
        print("prompt data", prompt_data.shape)
        print(prompt_data)
        print("attention input")
        print(prob_tensors.attention_input)

    # Run inference.
    decoder_z, decoder_t, encoder_activations, decoder_activations = state.apply_fn(
        state.params, prob_tensors.attention_input, prompt_data
    )

    activation_text = print_activations(
        n_examples, prompt_data, decoder_t, encoder_activations, decoder_activations, silence_print
    )

    file_path = os.path.join(folder_name, activations_file_name)
    with open(file_path, "w") as f:
        print("saving activations to", file_path)
        f.write(activation_text)

    return decoder_t


def count_params(params):
    leaves, _ = tree_flatten(params)
    return sum(leaf.size for leaf in leaves)


def inference_and_plot(
    state,
    prob_tensors,
    grammar,
    key,
    args,
    data,
    seed,
    n_epochs,
    epoch,
    loss,
    plot_file_name,
    activations_file_name,
    silence_print=False,
    folder_name=None
):
    decoder_t = run_and_print_inference(
        state=state,
        prob_tensors=prob_tensors,
        args=args,
        activations_file_name=activations_file_name,
        folder_name=folder_name,
        silence_print=silence_print,
    )
    text = ""
    text += f"decoder_t {decoder_t.shape}\n"

    temperature = 1
    generated_sentences = []
    for example_idx, example in enumerate(
        data.raw_inference_text.replace(" .", ".").split(".")
    ):
        if not example:
            continue
        text += f"Example {example_idx}: {example.capitalize()}\n"
        single_decoder_t = decoder_t[example_idx]
        for sample_idx in range(args.num_samples):
            key, subkey = jax.random.split(key)
            samples = sample(subkey, single_decoder_t, temperature=temperature)
            generated_sentence = " ".join([data.vocab[s] for s in samples]).capitalize()
            text += f"{generated_sentence}\n"
            generated_sentences.append(generated_sentence)
        text += "\n"
    text += f"seed: {seed}\n"

    # Generate histograms:
    training_sentences = [t.capitalize() for t in data.training_sentences]
    text += f"loss: {loss:.20e}\n"
    if not silence_print:
        print(text)

    histogram_results(
        training_sentences,
        generated_sentences,
        grammar=grammar,
        subtitle=f"seed: {seed}, total epochs: {n_epochs}, epoch: {epoch}, loss: {loss:.10e}",
        plot_file_name=plot_file_name,
        folder=folder_name,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Model arguments")
    parser.add_argument(
        "training_text", type=pathlib.Path
    )  # A text file of sentences to train on
    parser.add_argument(
        "--prompt_text", type=pathlib.Path
    )  # A text file of sentences to run inference on
    # if doing perturbation test, you also need to give it training_text
    parser.add_argument("--initialize_weights", action="store_true")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--lock_all_weights", action="store_true")
    parser.add_argument("--noise_value", type=float, default=0.01)
    parser.add_argument("--perturb_position", type=float, default=None)
    parser.add_argument("--perturb_token", type=float, default=None)
    parser.add_argument("--perturb_attention", type=float, default=None)
    parser.add_argument("--surgical_perturb", action="store_true", default=False)

    parser.add_argument("--layer_width", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--zero_out_right_weights", action="store_true")
    parser.add_argument("--zero_out_left_weights", action="store_true")
    parser.add_argument("--loss_threshold", type=float, default=None)
    parser.add_argument("--catsanddogs", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    # Parse args.
    args = parse_args()
    # If doing perturbation test, you also need to give it training_text
    if args.initialize_weights:
        assert args.training_text

    # Initialize RNG state.
    np_rng = np.random.default_rng()
    if args.seed:
        seed = args.seed
    else:
        seed = np_rng.integers(0, 2**32 - 1)

    num_epochs = args.num_epochs
    noise_value = args.noise_value

    key = jax.random.PRNGKey(seed)
    print(f"seed: {seed}\n")

    key, subkey = jax.random.split(key)
    noise_seed = jax.random.randint(
        subkey, (1,), jnp.iinfo(jnp.int32).min, jnp.iinfo(jnp.int32).max
    )[0]
    noise_seed = None  # type: ignore  # To not include noise in the training process.

    # Load training data.
    data = InputData(args.training_text, args.prompt_text, print_vals=False)

    # Construct the grammar.
    if args.catsanddogs:
        grammar = make_cat_dog_anavan()
    else:
        grammar = make_cat_dog_worm_bird_anavan()

    # Verify that the training sentences are valid.
    for sentence in data.training_sentences:
        assert grammar.is_valid(sentence), f"training data contains an invalid sentence: {sentence}"

    # Construct the probability tensors.
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
    state, model, lr = create_train_state(
        subkey,
        vocab=data.vocab,
        num_positions=prob_tensors.num_positions,
        layer_width=args.layer_width,
        num_layers=args.num_layers,
        noise_seed=noise_seed,
        initialize_weights=args.initialize_weights,
        perturb_flag=args.perturb,
        perturb_position=args.perturb_position,
        perturb_token=args.perturb_token,
        perturb_attention=args.perturb_attention,
        surgical_perturb=args.surgical_perturb,
        lock_all_weights=args.lock_all_weights,
        noise_value=noise_value,
        zero_out_right_weights=args.zero_out_right_weights,
        zero_out_left_weights=args.zero_out_left_weights,
        catsanddogs=args.catsanddogs,
    )

    # Check the initial loss.
    grads, loss = apply_model(
        state, prob_tensors.attention_input, all_t_tensors, all_outputs
    )
    print(f"initial loss: {loss:.20e}")

    # temp: duplicate our training data
    all_t_tensors = jnp.concatenate([all_t_tensors] * 100, axis=0)
    all_outputs = jnp.concatenate([all_outputs] * 100, axis=0)
    print(f"num training examples (padded): {all_t_tensors.shape[0]}")

    # Train the model.
    if args.training_text:
        n_epochs = num_epochs
        batch_size = 10
        n_steps_per_epoch = all_t_tensors.shape[0] // batch_size
        print_every = 10
        print(f"Training plan: {n_epochs} epochs, {n_steps_per_epoch} steps per epoch")
        key, subkey = jax.random.split(key)
    else:
        n_epochs = 0

    # Create a folder named {seed}_seed_{n_epochs}_num_epochs if it doesn't exist
    file_prefix = f"{seed}_seed_{n_epochs}_num_epochs_{noise_value}_noise_value_"
    folder_name = file_prefix
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    epoch = 0
    for epoch in range(n_epochs):
        # Shuffle the data.
        # shuffle_key = jax.random.fold_in(subkey, epoch)
        # all_t_tensors = jax.random.permutation(shuffle_key, all_t_tensors)
        # all_outputs = jax.random.permutation(shuffle_key, all_outputs)

        if epoch % print_every == 0 or epoch == n_epochs - 1:
            print("\nTop:", "↓" * 100)
            print(f"epoch {epoch}, loss: {loss:.20e}")
            printed_weights = print_params(state, data.vocab, silence_print=True)
            file_name = file_prefix + f"{epoch}_epoch_output_weights.txt"
            file_path = os.path.join(folder_name, file_name)
            with open(file_path, "w") as f:
                print("saving weights to", file_path)
                f.write(printed_weights)
            inference_and_plot(
                state=state,
                prob_tensors=prob_tensors,
                grammar=grammar,
                key=key,
                args=args,
                data=data,
                seed=seed,
                n_epochs=n_epochs,
                epoch=epoch,
                loss=loss,
                plot_file_name=file_prefix + f"{epoch}_epoch_output_histograms.png",
                activations_file_name=file_prefix + f"{epoch}_epoch_output_activations.txt",
                silence_print=False,
                folder_name=folder_name,
            )
            print("Bottom", "↑" * 100)

        if args.loss_threshold and loss < args.loss_threshold:
            break

        for step_idx in range(0, n_steps_per_epoch):
            start = step_idx * batch_size
            batch_input_data = all_t_tensors[start: start + batch_size]
            batch_output_data = all_outputs[
                step_idx * batch_size: (step_idx + 1) * batch_size
            ]
            grads, loss = apply_model(
                state, prob_tensors.attention_input, batch_input_data, batch_output_data
            )
            state = update_model(state, grads)
            if args.loss_threshold and loss < args.loss_threshold:
                break

    if n_epochs == 0:
        print("\nTop:", "↓" * 100)
        print("No training was done.")
        print(f"epoch {epoch}, loss: {loss:.20e}")
        # Print trained weights.
        printed_weights = print_params(state, data.vocab)
        # save printed weights to a file
        file_name = file_prefix + f"{epoch}_epoch_output_weights.txt"
        file_path = os.path.join(folder_name, file_name)
        with open(file_path, "w") as f:
            print("saving weights to", file_path)
            f.write(printed_weights)

        if not args.prompt_text:
            print("No prompt text given, exiting.")
            exit()

        inference_and_plot(
            state=state,
            prob_tensors=prob_tensors,
            grammar=grammar,
            key=key,
            args=args,
            data=data,
            seed=seed,
            n_epochs=n_epochs,
            epoch=epoch,
            loss=loss,
            plot_file_name=file_prefix + f"{epoch}_epoch_output_histograms.png",
            activations_file_name=file_prefix + f"{epoch}_epoch_output_activations.txt",
            folder_name=folder_name,
        )

    return seed, loss, lr


if __name__ == "__main__":
    for i in range(1):
        seed, loss, lr = main()
        if loss < 1e-3:
            # Save seed and loss to a file
            # Append to the file if it already exists
            with open("seed_loss.txt", "a") as f:
                f.write(f"seed: {seed}, loss: {loss:.20e}, learning rate: {lr}\n")
