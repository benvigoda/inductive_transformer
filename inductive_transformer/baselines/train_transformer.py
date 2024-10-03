from flax.training import train_state
from functools import partial
from pathlib import Path
import click
import jax
import jax.numpy as jnp
import numpy as np
import optax

from inductive_transformer.datasets.big_cat_small_dog import BigCatSmallDog
from inductive_transformer.datasets.anavan import make_cat_dog_worm_bird_anavan
from inductive_transformer.baselines.tokens import (
    load_dataset,
    make_dataset_from_sentences,
)
from inductive_transformer.baselines.transformer import (
    TransformerClassifier,
    make_causal_attention_mask,
)
from inductive_transformer.baselines.histograms import (
    SampleStatus,
    sample_status_names,
    generate_histogram_data,
    plot_histogram,
)
from inductive_transformer.jax_transformer.histogram_generations import (
    histogram_results,
)


# This file trains a transformer to output a probability distributions over tokens for each position
# in a sentence. For training and inference, we use an attention mask to ensure that the
# distribution at each position only depends on tokens before that position. During training, we use
# a cross entropy loss at each position, so the model is incentivized to generate the distribution
# over tokens conditional on all prior tokens. During inference, this network can be operated in an
# autoregressive manner by sampling only from the distribution for the next token, feeding in the
# result, and repeating.

# TODO
# - [x] Generate all valid sentences
# - [ ] epochs
# - [ ] For testing, I should use each sentence in the training and global sets once (not sampled)
# - [ ] I should also save stats on the number of in sample, out of sample, and invalid sentences
#   - [ ] report results from 10k samples?


def make_train_state(key, model, sequence_length, learning_rate):
    x = jnp.zeros((1, sequence_length), dtype=jnp.int32)
    key, subkey = jax.random.split(key)
    params = model.init(subkey, x)
    optimizer = optax.adam(learning_rate)
    print(model.tabulate(key, x, compute_flops=True, compute_vjp_flops=True))
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


def sample_sentences(key, data, batch_size):
    # Generates a batch of sentences.
    n_sentences, sequence_length = data.shape
    sentence_ids = jax.random.randint(key, (batch_size,), 0, n_sentences)
    result = data[sentence_ids]
    assert result.shape == (batch_size, sequence_length)
    return result


def word_ids_to_one_hot(data, vocab_size):
    return jax.nn.one_hot(data, vocab_size)


def generate_batch(key, data, batch_size):
    sequence_length = data.shape[1]
    key, subkey = jax.random.split(key)
    data = sample_sentences(subkey, data, batch_size)
    assert data.shape == (batch_size, sequence_length)
    return data


@jax.jit
def evaluate_step(data, state):
    sequence_length = data.shape[1]
    mask = make_causal_attention_mask(sequence_length)
    logits = state.apply_fn(state.params, data, mask=mask, training=False)
    x_entropies = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits[:, 0:-1, :], labels=data[:, 1:]
    )
    return x_entropies.mean()


def evaluate(dataset, state, batch_size):
    n_sentences = dataset.shape[0]
    n_batches = n_sentences // batch_size
    assert n_batches * batch_size == n_sentences, "Expected even batches"
    losses = []
    for batch in range(n_batches):
        batch_data = dataset[batch * batch_size : (batch + 1) * batch_size]
        mean_x_entropy = evaluate_step(batch_data, state)
        losses.append(mean_x_entropy)
    return np.mean(np.array(losses))


@partial(jax.jit, static_argnums=(3))
def train_step(dropout_key, batch_x, state, vocab_size):
    batch_size, sequence_length = batch_x.shape
    mask = make_causal_attention_mask(sequence_length)
    step_dropout_key = jax.random.fold_in(dropout_key, state.step)

    # Define the loss function.
    def loss_fn(params):
        logits = state.apply_fn(
            params,
            batch_x,
            mask=mask,
            # training=True,
            training=False,
            rngs={"dropout": step_dropout_key},
        )
        assert logits.shape == (batch_size, sequence_length, vocab_size)
        x_entropies = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits[:, 0:-1, :], labels=batch_x[:, 1:]
        )
        assert x_entropies.shape == (batch_size, sequence_length - 1)
        return x_entropies.mean()

    # Compute gradients and apply updates.
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train(
    key,
    data,
    train_data,
    test_data,
    model,
    state,
    batch_size,
    n_epochs,
    vocab_size,
    use_start_token,
    classify_sentence,
):
    n_sentences = train_data.shape[0]

    train_batch_size = batch_size if batch_size < n_sentences else n_sentences
    n_batches_per_epoch = train_data.shape[0] // train_batch_size
    assert (
        n_batches_per_epoch * train_batch_size == train_data.shape[0]
    ), "Expected even batches"

    assert batch_size < test_data.shape[0]
    n_batches_per_epoch_test = test_data.shape[0] // batch_size
    assert (
        n_batches_per_epoch_test * batch_size == test_data.shape[0]
    ), "Expected even batches"

    key, dropout_key = jax.random.split(key)

    losses = np.zeros((n_epochs, 2), dtype=np.float32)
    sample_stats = np.zeros((n_epochs, 3), dtype=np.int32)
    for epoch in range(n_epochs):
        # shuffle the dataset
        if train_batch_size < n_sentences:
            key, subkey = jax.random.split(key)
            indices = jax.random.permutation(subkey, jnp.arange(train_data.shape[0]))
        else:
            indices = jnp.arange(train_data.shape[0])

        # run the training loop
        for step in range(n_batches_per_epoch):
            batch_indices = indices[
                step * train_batch_size : (step + 1) * train_batch_size
            ]
            batch_x = train_data[batch_indices]
            state, loss = train_step(
                dropout_key,
                batch_x,
                state,
                vocab_size,
            )
            # if step % 100 == 0 or step == n_batches_per_epoch - 1:
            #     print(f"Epoch {epoch}, step {step}, loss: {loss}")

        # Evaluate on the training and test data.
        train_loss = evaluate(train_data, state, train_batch_size)
        test_loss = evaluate(test_data, state, batch_size)

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            key, subkey = jax.random.split(key)
            _, n_in_sample, n_out_of_sample, n_invalid = sample(
                subkey,
                model,
                state.params,
                data,
                train_data,
                vocab_size,
                use_start_token,
                classify_sentence,
            )
        else:
            n_in_sample = 0
            n_out_of_sample = 0
            n_invalid = 0

        losses[epoch] = [train_loss, test_loss]
        sample_stats[epoch] = [n_in_sample, n_out_of_sample, n_invalid]
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(
                f"Epoch {epoch}, train loss: {train_loss}, test loss: {test_loss}, n_in_sample: {n_in_sample}, n_out_of_sample: {n_out_of_sample}, n_invalid: {n_invalid}"
            )

    return state, losses, sample_stats


def sample(
    key, model, params, data, dataset, n_classes, use_start_token, classify_sentence
):
    sequence_length = dataset.shape[1]
    mask = make_causal_attention_mask(sequence_length)

    generated_sentences = []
    n_samples = 1000
    n_in_sample = 0
    n_out_of_sample = 0
    n_invalid = 0

    for sample_batch in range(10):
        key, subkey = jax.random.split(key)
        # We will only let the model see the first column of these samples. So if we're prepending start
        # tokens, there's no information. If we're not, it gets single word prompts.
        generated_tokens = generate_batch(subkey, dataset, n_samples)
        for position in range(sequence_length - 1):
            logits = model.apply(params, generated_tokens, mask=mask, training=False)
            assert logits.shape == (n_samples, sequence_length, n_classes)

            # This samples according to the categorical distribution given by logits.
            key, subkey = jax.random.split(key)
            next_tokens = jax.random.categorical(subkey, logits[:, position, :])
            assert next_tokens.shape == (n_samples,)

            # This chooses the most likely word.
            # next_tokens = jnp.argmax(jax.nn.softmax(logits[:, position, :]), axis=-1)

            generated_tokens = generated_tokens.at[:, position + 1].set(next_tokens)

        if use_start_token:
            generated_tokens = generated_tokens[:, 1:]

        new_sentences = data.ids_to_strings(generated_tokens)
        generated_sentences.extend(new_sentences)

        for id in range(n_samples):
            category = classify_sentence(new_sentences[id])
            if category == SampleStatus.IN_SAMPLE:
                n_in_sample += 1
            elif category == SampleStatus.OUT_OF_SAMPLE:
                n_out_of_sample += 1
            else:
                n_invalid += 1

    return generated_sentences, n_in_sample, n_out_of_sample, n_invalid


@click.command()
@click.option("--training_sentences", "-i", type=click.Path(exists=True), required=True)
@click.option("--learning_rate", "-l", type=float, default=1e-4)
@click.option("--n_epochs", "-e", type=int, default=10000)
@click.option(
    "--use_start_token",
    "-s",
    is_flag=True,
    help="Prepend a start token so the model tries to guess the first word.",
)
def main(training_sentences, learning_rate, n_epochs, use_start_token):
    np_rng = np.random.default_rng()
    seed = np_rng.integers(0, 2**32 - 1)
    print(f"seed: {seed}\n")
    key = jax.random.PRNGKey(seed)

    # Text file pipeline
    print(f"Loading training data from {training_sentences}...")
    data = load_dataset(training_sentences)
    print(f"Loaded {data.n_sentences} sentences.")
    print(f"Vocabulary size: {data.vocab_size}")
    print(f"Sentence length: {data.sentence_length}")

    # Verify that the training sentences are valid.
    grammar = make_cat_dog_worm_bird_anavan()
    training_sentence_strings = data.ids_to_strings(data.data)
    training_sentences_set = set(training_sentence_strings)
    for sentence in training_sentence_strings:
        # print(sentence)
        assert grammar.is_valid(
            sentence
        ), f"training data contains an invalid sentence: {sentence}"

    dataset = data.data
    assert dataset.shape == (data.n_sentences, data.sentence_length)
    if use_start_token:
        start_tokens = jnp.full(data.n_sentences, data.blank_token).reshape(-1, 1)
        dataset = jnp.concatenate([start_tokens, dataset], axis=-1)

    # Generate all valid sentences (for testing purposes)
    all_valid_sentences = grammar.generate()
    all_valid_sentences = data.strings_to_ids(all_valid_sentences)

    # Generative pipeline
    """
    print("Generating data...")
    grammar = BigCatSmallDog()
    # This is a list of lists of words (strings).
    all_valid_sentences = grammar.all_valid_sentences()
    n_total_sentences = len(all_valid_sentences)
    all_valid_sentences_set = set(" ".join(words) for words in all_valid_sentences)
    print(f"Generated {n_total_sentences} sentences.")

    Extract a subset of sentences to train on.
    Note: this is useful if we generate all possible valid sentences and choose a subset.
    n_training_sentences = int(0.75 * n_total_sentences)
    sentence_indices = jnp.arange(n_total_sentences)
    key, subkey = jax.random.split(key)
    sentence_indices = jax.random.permutation(subkey, sentence_indices)
    training_sentences = [
        all_valid_sentences[i] for i in sentence_indices[:n_training_sentences]
    ]
    training_sentences_set = set(" ".join(words) for words in training_sentences)

    # Build the dataset.
    data = make_dataset_from_sentences(training_sentences)
    print(
        f"Training on {data.n_sentences} sentences of length {data.sentence_length} "
        f"with a vocabulary size of {data.vocab_size}."
    )
    print("")
    """

    # Fix hyperparameters.
    # The sentence length is one longer than the length of the sentences in the dataset because we
    # include a start token at the beginning of each sentence (so the model can predict the first
    # word).
    sentence_length = data.sentence_length
    sequence_length = sentence_length if not use_start_token else sentence_length + 1
    n_classes = data.vocab_size
    embedding_dim = 16
    feedforward_dim = 4 * embedding_dim
    n_blocks = 2
    n_heads = 2
    k_dim = 4
    v_dim = 4
    dropout_rate = 0.1

    batch_size = 64  # this will be reduced if the dataset is smaller
    learning_rate = 1e-3

    stem = Path(training_sentences).stem
    log_file_name = f"log_{stem}.txt"
    log_file = open(log_file_name, "w")

    # write the hyperparameters to the log file
    log_file.write(f"dataset: {training_sentences}\n")
    log_file.write(f"seed: {seed}\n")
    log_file.write(f"sequence_length: {sequence_length}\n")
    log_file.write(f"sentence_length: {sentence_length}\n")
    log_file.write(f"n_classes: {n_classes}\n")
    log_file.write(f"embedding_dim: {embedding_dim}\n")
    log_file.write(f"feedforward_dim: {feedforward_dim}\n")
    log_file.write(f"n_blocks: {n_blocks}\n")
    log_file.write(f"n_heads: {n_heads}\n")
    log_file.write(f"k_dim: {k_dim}\n")
    log_file.write(f"v_dim: {v_dim}\n")
    # log_file.write(f"dropout_rate: {dropout_rate}\n")
    log_file.write(f"batch_size: {batch_size}\n")
    log_file.write(f"n_epochs: {n_epochs}\n")
    log_file.write(f"learning_rate: {learning_rate}\n")
    log_file.write("\n")

    print("")
    print("Initializing model...")
    key, subkey = jax.random.split(key)
    model = TransformerClassifier(
        sequence_length=sequence_length,
        n_classes=n_classes,
        embedding_dim=embedding_dim,
        feedforward_dim=feedforward_dim,
        n_blocks=n_blocks,
        n_heads=n_heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dropout_rate=dropout_rate,
    )
    train_state = make_train_state(
        subkey,
        model,
        sequence_length,
        learning_rate,
    )
    print("")

    # Text file pipeline
    def classify_sentence(sentence: str) -> SampleStatus:
        if sentence in training_sentences_set:
            return SampleStatus.IN_SAMPLE
        if grammar.is_valid(sentence):
            return SampleStatus.OUT_OF_SAMPLE
        return SampleStatus.INVALID

    print("Training...")
    key, subkey = jax.random.split(key)
    state, losses, sample_stats = train(
        subkey,
        data,
        dataset,
        all_valid_sentences,
        model,
        train_state,
        batch_size,
        n_epochs,
        data.vocab_size,
        use_start_token,
        classify_sentence,
    )
    log_file.write(
        "epoch, train_loss, test_loss, n_in_sample, n_out_of_sample, n_invalid\n"
    )
    for epoch, (
        (train_loss, test_loss),
        (n_in_sample, n_out_of_sample, n_invalid),
    ) in enumerate(zip(losses, sample_stats)):
        log_file.write(
            f"{epoch} {train_loss} {test_loss} {n_in_sample} {n_out_of_sample} {n_invalid}\n"
        )
    log_file.write("\n")
    print("")

    print("Sampling...")
    # def sample(key, model, params, data, dataset, n_classes, use_start_token, classify_sentence):
    key, subkey = jax.random.split(key)
    generated_sentences, n_in_sample, n_out_of_sample, n_invalid = sample(
        subkey,
        model,
        state.params,
        data,
        dataset,
        n_classes,
        use_start_token,
        classify_sentence,
    )
    log_file.write(f"n_in_sample: {n_in_sample}\n")
    log_file.write(f"n_out_of_sample: {n_out_of_sample}\n")
    log_file.write(f"n_invalid: {n_invalid}\n")
    log_file.close()

    n_printed_samples = 50
    for id in range(n_printed_samples):
        category = classify_sentence(generated_sentences[id])
        print(f"{generated_sentences[id]} ({sample_status_names[category]})")
    print("")

    print("Generating histograms...")
    histogram_data = generate_histogram_data(generated_sentences, classify_sentence)
    plot_histogram(histogram_data, "histogram_t.png", size=(8.0, 12.0))

    # histogram_results(
    #     training_sentence_strings,
    #     generated_sentences,
    #     grammar=grammar,
    #     subtitle=f"subtitle?",
    #     plot_file_name="histogram_t_v2.png",
    #     folder=None,
    # )


if __name__ == "__main__":
    main()
