from flax.training import train_state
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax

from grammars import BigCatSmallDog
from tokens import load_dataset, make_dataset_from_sentences
from models import FullyConnectedSeqToSeq
from histograms import (
    SampleStatus,
    sample_status_names,
    generate_histogram_data,
    plot_histogram,
)


# This file trains a fully connected network to output a probability distribution over tokens for
# each position in a sentence. During training, words in the input sentence are randomly replaced
# with a BLANK token. Each output distribution is allowed to see all input tokens, including
# non-blanks that occur after the output position in question. So if only one word is blanked out,
# the network is trained to output the distribution conditioned on all other words (before and
# after). If all words are replaced with blanks, the network will try to generate marginal
# distributions for all output positions. (Of course if you sample from these distributions
# independently, the tokens in different positions will not be correlated.)


def make_train_state(key, model, dataset, learning_rate):
    x = jnp.zeros((1, dataset.sentence_length, dataset.vocab_size))
    key, subkey = jax.random.split(key)
    params = model.init(subkey, x)
    optimizer = optax.adam(learning_rate)
    print(model.tabulate(key, x, compute_flops=True, compute_vjp_flops=True))
    return model, train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )


def sample_sentences(key, data, batch_size):
    # Generates a batch of sentences.
    n_sentences, sentence_length = data.shape
    sentence_ids = jax.random.randint(key, (batch_size,), 0, n_sentences)
    result = data[sentence_ids]
    assert result.shape == (batch_size, sentence_length)
    return result


def blank_out_words(key, data, p, vocab_size):
    # Randomly blank out some words.
    # Data should be of shape (..., batch_size, sentence_length).
    blank_token = vocab_size - 1
    mask = jax.random.bernoulli(key, p, data.shape)
    blanked_data = jnp.where(mask, blank_token, data)
    return blanked_data, mask


def word_ids_to_one_hot(data, vocab_size):
    return jax.nn.one_hot(data, vocab_size)


def generate_batch(key, data, vocab_size, batch_size):
    sentence_length = data.shape[1]
    key, subkey = jax.random.split(key)
    data = sample_sentences(subkey, data, batch_size)
    assert data.shape == (batch_size, sentence_length)
    p = 1.0 / sentence_length
    masked_data, mask = blank_out_words(key, data, p, vocab_size)
    assert masked_data.shape == (batch_size, sentence_length)
    return masked_data, data, mask


def train_step(state, batch_x, batch_y):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn(params, batch_x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch_y
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jax.jit, static_argnums=(3, 4))
def full_train_step(key, data, state, vocab_size, batch_size):
    step_key = jax.random.fold_in(key, state.step)
    batch_x, batch_y, _ = generate_batch(step_key, data, vocab_size, batch_size)
    batch_x = word_ids_to_one_hot(batch_x, vocab_size)
    return train_step(state, batch_x, batch_y)


def train(key, dataset, state, batch_size, n_steps):
    for step in range(n_steps):
        state, loss = full_train_step(
            key, dataset.data, state, dataset.vocab_size, batch_size
        )
        if step % 1000 == 0:
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
    n_training_sentences = int(0.5 * n_total_sentences)
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

    print("Initializing model...")
    learning_rate = 2e-5
    sentence_words = data.sentence_length * data.vocab_size
    layers = [
        4 * sentence_words,
        sentence_words,
        4 * sentence_words,
    ]
    key, subkey = jax.random.split(key)
    model, train_state = make_train_state(
        subkey,
        FullyConnectedSeqToSeq(layers=layers),
        data,
        learning_rate,
    )
    key, subkey = jax.random.split(key)
    print("")

    print("Training...")
    batch_size = 256
    n_steps = 10000
    key, subkey = jax.random.split(key)
    state = train(subkey, data, train_state, batch_size, n_steps)
    print("")

    print("Sampling...")
    n_samples = 10000
    key, subkey = jax.random.split(key)
    sample_x_ids, sample_y_ids, mask = generate_batch(
        subkey, data.data, data.vocab_size, n_samples
    )
    assert sample_x_ids.shape == (n_samples, data.sentence_length)
    assert sample_y_ids.shape == sample_x_ids.shape
    assert mask.shape == sample_x_ids.shape
    sample_x_words = data.ids_to_strings(sample_x_ids)
    sample_y_words = data.ids_to_strings(sample_y_ids)
    sample_x_one_hot = word_ids_to_one_hot(sample_x_ids, data.vocab_size)
    log_probs = state.apply_fn(state.params, sample_x_one_hot)
    key, subkey = jax.random.split(key)

    # This samples according to the categorical distribution given by log_probs.
    generated_ids = jax.random.categorical(subkey, log_probs)

    # This chooses the most likely word.
    # probs = jax.nn.softmax(log_probs)
    # generated_ids = jnp.argmax(probs, axis=-1)

    assert generated_ids.shape == (n_samples, data.sentence_length)
    generated_words = data.ids_to_strings(generated_ids)

    def classify_sentence(sentence: str) -> SampleStatus:
        if sentence in training_sentences_set:
            return SampleStatus.IN_SAMPLE
        if sentence in all_valid_sentences_set:
            return SampleStatus.OUT_OF_SAMPLE
        return SampleStatus.INVALID

    n_printed_samples = 50
    limit = min(n_printed_samples, n_samples)
    for id in range(limit):
        category = classify_sentence(generated_words[id])
        print(
            sample_y_words[id],
            "=>",
            sample_x_words[id],
            "=>",
            generated_words[id],
            sample_status_names[category],
        )
    print("")

    print("Generating histograms...")
    histogram_data = generate_histogram_data(generated_words, classify_sentence)
    plot_histogram(histogram_data, "histogram.png")


if __name__ == "__main__":
    main()
