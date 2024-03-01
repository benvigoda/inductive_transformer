import jax
import jax.numpy as jnp
import time


def custom_normalize(tensor, dim=0, default_constant=0.5):
    '''
    dim is the dimension on which to normalize
    default_constant is the value to use when the sum is zero
    '''
    # Compute the sum along dim=dim and keepdim=True to maintain the dimensions for broadcasting
    sum_tensor = jnp.sum(tensor, axis=dim, keepdims=True)

    # Create a mask where the sum is zero
    mask = sum_tensor == 0

    # Replace zero sums with ones to avoid division by zero and then divide
    result = tensor / jnp.where(mask, jnp.ones_like(sum_tensor), sum_tensor)

    # Where the sum was zero, replace with the constant C
    result = jnp.where(mask, jnp.full_like(result, fill_value=default_constant), result)

    return result


custom_normalize_vmap = jax.vmap(custom_normalize, in_axes=(0))

custom_normalize = jax.jit(custom_normalize)
custom_normalize_vmap = jax.jit(custom_normalize_vmap)


if __name__ == "__main__":
    n_iterations = 1_000_000

    # python for loop
    """
    tensor = jnp.ones((2, 2, 10, 2))
    # custom_normalize(tensor) # run it once to compile the function
    start = time.time()
    for _ in range(n_iterations):
        tensor = custom_normalize(tensor)
    end = time.time()
    print("python for loop:", end - start)
    """

    # vmap
    print("making tensor")
    tensor = jnp.ones((n_iterations, 2, 2, 10, 2))
    print("compiling")
    custom_normalize_vmap(tensor) # run it once to compile the function
    print("running")
    start = time.time()
    tensor = custom_normalize_vmap(tensor)
    end = time.time()
    print("vmap:", end - start)

    # jax for loop
    """
    @jax.jit
    def custom_normalize_for_loop(tensor):
        return jax.lax.fori_loop(0, n_iterations, lambda i, x: custom_normalize(x), tensor)

    tensor = jnp.ones((2, 2, 10, 2))
    # custom_normalize_for_loop(tensor) # run it once to compile the function
    start = time.time()
    tensor = custom_normalize_for_loop(tensor)
    end = time.time()
    print("jax for loop:", end - start)
    """
