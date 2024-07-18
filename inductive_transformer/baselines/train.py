from flax.training import train_state
from functools import partial
import click
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tokens import load_dataset
from models import FullyConnected


def make_train_state(key, model, dataset, learning_rate):
    x = jnp.zeros((1, dataset.sentence_length, dataset.vocab_size))
    params = model.init(key, x)
    optimizer = optax.adam(learning_rate)
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
    masked_data = word_ids_to_one_hot(masked_data, vocab_size)
    assert masked_data.shape == (batch_size, sentence_length, vocab_size)
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
    return train_step(state, batch_x, batch_y)


def train(key, dataset, state, batch_size, n_steps):
    for step in range(n_steps):
        state, loss = full_train_step(
            key, dataset.data, state, dataset.vocab_size, batch_size
        )
        if step % 1000 == 0:
            print(f"Step {step}, loss: {loss}")
    return state


@click.command()
@click.option("--dataset", "-d", type=click.Path(exists=True), required=True)
def main(dataset):
    np_rng = np.random.default_rng()
    seed = np_rng.integers(0, 2**32 - 1)
    print(f"seed: {seed}\n")
    key = jax.random.PRNGKey(seed)

    print("Loading data...")
    data = load_dataset(dataset)
    print(
        f"Loaded {data.n_sentences} sentences of length {data.sentence_length} "
        "with a vocabulary size of {data.vocab_size}."
    )
    print("")

    print("Initializing model...")
    learning_rate = 1e-4
    sentence_words = data.sentence_length * data.vocab_size
    layers = [
        4 * sentence_words,
        3 * sentence_words,
        2 * sentence_words,
        sentence_words,
    ]
    key, subkey = jax.random.split(key)
    model, train_state = make_train_state(
        subkey, FullyConnected(layers=layers), data, learning_rate
    )
    print("")

    print("Training...")
    batch_size = 256
    n_steps = 20000
    key, subkey = jax.random.split(key)
    state = train(subkey, data, train_state, batch_size, n_steps)


if __name__ == "__main__":
    main()
