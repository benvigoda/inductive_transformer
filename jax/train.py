from flax.training import train_state
from pprint import pprint
import jax
import jax.numpy as jnp
import optax

from model import InductiveTransformer


def create_train_state(key):
    """Creates initial `TrainState`."""
    bernoulli_width = 2
    num_positions = 2
    vocab_size = 4
    layer_width = 2
    num_layers = 2

    model = InductiveTransformer(
        layer_width=layer_width,
        num_positions=num_positions,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )

    key, subkey_0, subkey_1, subkey_2 = jax.random.split(key, 4)
    z_in = jax.random.normal(subkey_0, (bernoulli_width, layer_width))
    t_in = jax.random.normal(subkey_1, (num_layers, num_positions, vocab_size, layer_width))
    params = model.init(subkey_2, z_in, t_in)
    # TODO We can manipulate our initial params here.
    pprint(params)

    tx = optax.adam(learning_rate=1.0e-3)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx),


@jax.jit
def apply_model(state, z_in, t_in):
    """Computes gradients and loss for a single instance (not yet batched)."""

    def loss_fn(params):
        z_out, t_out = state.apply_fn({"params": params}, z_in, t_in)
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


if __name__ == "__main__":
    # Initialize RNG state.
    np_rng = np.random.default_rng()
    seed = np_rng.integers(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    print(f"seed: {seed}\n")

    key, subkey = jax.random.split(key)
    state = create_train_state(subkey)
    print(state)
    # print(apply_model(state, b_measurements, magnet_positions))

    # Train the model.
    n_training_steps = 100
    for step in range(0, n_training_steps):
        key = jax.random.fold_in(key, step)
        state, loss = train_step(base_positions, state, key)
        smoothed_loss = loss_smoothing * smoothed_loss + (1 - loss_smoothing) * loss
        if step % print_every == 0:
            print(f"step {step}, loss: {loss:.3e}, smoothed loss: {smoothed_loss:.3e}, lr: {lr_schedule(step):.3e}")
