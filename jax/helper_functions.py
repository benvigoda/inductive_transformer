import jax.numpy as jnp


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
