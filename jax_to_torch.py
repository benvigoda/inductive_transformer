import jax
import jax.numpy as jnp
import numpy as np
# from jax_transformer.train import create_train_state

if __name__ == "__main__":
    # Initialize RNG state.
    np_rng = np.random.default_rng()
    seed = np_rng.integers(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    print(f"seed: {seed}\n")
