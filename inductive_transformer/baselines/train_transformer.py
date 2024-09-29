from flax.training import train_state
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax

from grammars import BigCatSmallDog
from tokens import load_dataset, make_dataset_from_sentences
from transformer import TransformerClassifier, make_causal_attention_mask
from histograms import (
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


def make_train_state(key, model, dataset, learning_rate):
    x = jnp.zeros((1, dataset.sentence_length), dtype=jnp.int32)
    key, subkey = jax.random.split(key)
    params = model.init(subkey, x)
    optimizer = optax.adam(learning_rate)
    print(model.tabulate(key, x, compute_flops=True, compute_vjp_flops=True))
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


def sample_sentences(key, data, batch_size):
    # Generates a batch of sentences.
    n_sentences, sentence_length = data.shape
    sentence_ids = jax.random.randint(key, (batch_size,), 0, n_sentences)
    result = data[sentence_ids]
    assert result.shape == (batch_size, sentence_length)
    return result


def word_ids_to_one_hot(data, vocab_size):
    return jax.nn.one_hot(data, vocab_size)


def generate_batch(key, data, batch_size, padding_token):
    sentence_length = data.shape[1]
    key, subkey = jax.random.split(key)
    data = sample_sentences(subkey, data, batch_size)
    assert data.shape == (batch_size, sentence_length)
    # We want our network to guess the first word as well. The easiest way to achieve this is to
    # make every sentence start with the padding token.
    start_tokens = jnp.full(batch_size, padding_token).reshape(-1, 1)
    data = jnp.concatenate([start_tokens, data], axis=-1)
    assert data.shape == (batch_size, sentence_length + 1)
    return data


@partial(jax.jit, static_argnums=(4, 5, 6))
def train_step(key, dropout_key, data, state, vocab_size, batch_size, padding_token):
    # Sample a batch of sentences.
    sentence_length = data.shape[-1] + 1
    step_key = jax.random.fold_in(key, state.step)
    step_dropout_key = jax.random.fold_in(dropout_key, state.step)
    batch_x = generate_batch(step_key, data, batch_size, padding_token)
    assert batch_x.shape == (batch_size, sentence_length)

    # Hopefully JAX/XLA recognizes that this is a constant.
    mask = make_causal_attention_mask(sentence_length)

    # Define the loss function.
    def loss_fn(params):
        logits = state.apply_fn(
            params,
            batch_x,
            mask=mask,
            training=True,
            rngs={"dropout": step_dropout_key},
        )
        assert logits.shape == (batch_size, sentence_length, vocab_size)
        x_entropies = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits[:, 0:-1, :], labels=batch_x[:, 1:]
        )
        assert x_entropies.shape == (batch_size, sentence_length - 1)
        return x_entropies.mean()

    # Compute gradients and apply updates.
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train(key, dropout_key, dataset, state, batch_size, n_steps, padding_token):
    for step in range(n_steps):
        state, loss = train_step(
            key,
            dropout_key,
            dataset.data,
            state,
            dataset.vocab_size,
            batch_size,
            padding_token,
        )
        if step % 1000 == 0:
            print(f"Step {step}, loss: {loss}")

    print(f"Step {step}, loss: {loss}")
    return state


# @click.command()
# @click.option("--dataset", "-d", type=click.Path(exists=True), required=True)
def main():
    np_rng = np.random.default_rng()
    seed = np_rng.integers(0, 2**32 - 1)
    print(f"seed: {seed}\n")
    key = jax.random.PRNGKey(seed)

    print("Generating data...")
    grammar = BigCatSmallDog()
    # This is a list of lists of words (strings).
    all_valid_sentences = grammar.all_valid_sentences()
    n_total_sentences = len(all_valid_sentences)
    all_valid_sentences_set = set(" ".join(words) for words in all_valid_sentences)
    print(f"Generated {n_total_sentences} sentences.")

    # Extract a subset of sentences to train on.
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

    # Fix hyperparameters.
    # The sentence length is one longer than the length of the sentences in the dataset because we
    # include a start token at the beginning of each sentence (so the model can predict the first
    # word).
    sequence_length = data.sentence_length + 1
    n_classes = data.vocab_size
    embedding_dim = 64
    feedforward_dim = 4 * embedding_dim
    n_blocks = 4
    n_heads = 2
    k_dim = embedding_dim // n_heads
    v_dim = embedding_dim // n_heads
    dropout_rate = 0.1

    batch_size = 8
    n_steps = 10000
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
        data,
        learning_rate,
    )
    print("")

    print("Training...")
    key, batch_key, dropout_key = jax.random.split(key, 3)
    padding_token = data.blank_token
    state = train(
        batch_key, dropout_key, data, train_state, batch_size, n_steps, padding_token
    )
    print("")
    exit(0)

    print("Sampling...")
    n_samples = 1000
    key, subkey = jax.random.split(key)
    prompts = generate_batch(subkey, data, n_samples)
    print(prompts)
    assert prompts.shape == (n_samples, data.sentence_length)
    responses = model.apply(state.params, prompts, training=False)

    sample_positions = jnp.zeros((n_samples,), dtype=jnp.int32)
    for position in range(data.sentence_length):
        logits = model.apply(
            state.params, word_ids_to_one_hot(generated_tokens, data.vocab_size)
        )

        # This samples according to the categorical distribution given by logits.
        key, subkey = jax.random.split(key)
        next_tokens = jax.random.categorical(subkey, logits)

        # This chooses the most likely word.
        # next_tokens = jnp.argmax(jax.nn.softmax(logits), axis=-1)

        generated_tokens = generated_tokens.at[
            jnp.arange(n_samples), sample_positions
        ].set(next_tokens, mode="drop")
        sample_positions = jnp.where(
            sample_positions < data.sentence_length,
            sample_positions + 1,
            sample_positions,
        )

    assert generated_tokens.shape == (n_samples, data.sentence_length)
    generated_sentences = data.ids_to_strings(generated_tokens)

    def classify_sentence(sentence: str) -> SampleStatus:
        if sentence in training_sentences_set:
            return SampleStatus.IN_SAMPLE
        if sentence in all_valid_sentences_set:
            return SampleStatus.OUT_OF_SAMPLE
        return SampleStatus.INVALID

    n_printed_samples = 50
    limit = min(n_printed_samples, n_samples)
    for id in range(limit):
        category = classify_sentence(generated_sentences[id])
        print(f"{generated_sentences[id]} ({sample_status_names[category]})")
    print("")

    print("Generating histograms...")
    histogram_data = generate_histogram_data(generated_sentences, classify_sentence)
    plot_histogram(histogram_data, "histogram.png", size=(8.0, 12.0))


if __name__ == "__main__":
    main()
