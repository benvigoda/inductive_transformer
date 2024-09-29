from flax.training import train_state
from functools import partial
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


# This file trains a transformer to output a probability distributions over tokens for each position
# in a sentence. For training and inference, we use an attention mask to ensure that the
# distribution at each position only depends on tokens before that position. During training, we use
# a cross entropy loss at each position, so the model is incentivized to generate the distribution
# over tokens conditional on all prior tokens. During inference, this network can be operated in an
# autoregressive manner by sampling only from the distribution for the next token, feeding in the
# result, and repeating.


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


@partial(jax.jit, static_argnums=(4, 5))
def train_step(key, dropout_key, data, state, vocab_size, batch_size):
    # Sample a batch of sentences.
    sequence_length = data.shape[-1]
    step_key = jax.random.fold_in(key, state.step)
    step_dropout_key = jax.random.fold_in(dropout_key, state.step)
    batch_x = generate_batch(step_key, data, batch_size)
    assert batch_x.shape == (batch_size, sequence_length)

    # Hopefully JAX/XLA recognizes that this is a constant.
    mask = make_causal_attention_mask(sequence_length)

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


def train(key, dropout_key, dataset, state, batch_size, n_steps, vocab_size):
    for step in range(n_steps):
        state, loss = train_step(
            key,
            dropout_key,
            dataset,
            state,
            vocab_size,
            batch_size,
        )
        if step % 1000 == 0:
            print(f"Step {step}, loss: {loss}")

    print(f"Step {step}, loss: {loss}")
    return state


@click.command()
@click.option("--training_sentences", "-t", type=click.Path(exists=True), required=True)
@click.option(
    "--use_start_token",
    "-s",
    is_flag=True,
    help="Prepend a start token so the model tries to guess the first word.",
)
def main(training_sentences, use_start_token):
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
    sentence_strings = data.ids_to_strings(data.data)
    training_sentences_set = set(sentence_strings)
    for sentence in sentence_strings:
        print(sentence)
        assert grammar.is_valid(
            sentence
        ), f"training data contains an invalid sentence: {sentence}"

    dataset = data.data
    assert dataset.shape == (data.n_sentences, data.sentence_length)
    if use_start_token:
        start_tokens = jnp.full(data.n_sentences, data.blank_token).reshape(-1, 1)
        dataset = jnp.concatenate([start_tokens, dataset], axis=-1)

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

    batch_size = 64
    n_steps = 50000
    learning_rate = 1e-5

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

    print("Training...")
    key, batch_key, dropout_key = jax.random.split(key, 3)
    state = train(
        batch_key,
        dropout_key,
        dataset,
        train_state,
        batch_size,
        n_steps,
        data.vocab_size,
    )
    print("")

    print("Sampling...")
    mask = make_causal_attention_mask(sequence_length)
    n_samples = 1000
    key, subkey = jax.random.split(key)
    # We will only let the model see the first column of these samples. So if we're prepending start
    # tokens, there's no information. If we're not, it gets single word prompts.
    generated_tokens = generate_batch(subkey, dataset, n_samples)
    for position in range(sequence_length - 1):
        logits = model.apply(state.params, generated_tokens, mask=mask, training=False)
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

    generated_sentences = data.ids_to_strings(generated_tokens)

    # Text file pipeline
    def classify_sentence(sentence: str) -> SampleStatus:
        if sentence in training_sentences_set:
            return SampleStatus.IN_SAMPLE
        if grammar.is_valid(sentence):
            return SampleStatus.OUT_OF_SAMPLE
        return SampleStatus.INVALID

    # Generative pipeline
    """
    def classify_sentence(sentence: str) -> SampleStatus:
        if sentence in training_sentences_set:
            return SampleStatus.IN_SAMPLE
        if sentence in all_valid_sentences_set:
            return SampleStatus.OUT_OF_SAMPLE
        return SampleStatus.INVALID
    """

    n_printed_samples = 50
    limit = min(n_printed_samples, n_samples)
    for id in range(limit):
        category = classify_sentence(generated_sentences[id])
        print(f"{generated_sentences[id]} ({sample_status_names[category]})")
    print("")

    print("Generating histograms...")
    histogram_data = generate_histogram_data(generated_sentences, classify_sentence)
    plot_histogram(histogram_data, "histogram_t.png", size=(8.0, 12.0))


if __name__ == "__main__":
    main()
