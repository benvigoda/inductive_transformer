import jax.numpy as jnp  # type: ignore
import jax  # type: ignore

EPSILON = 1e-20
IMPROBABLE = 1e-9
PROBABLE = 1 - IMPROBABLE


def get_num_layers(params: dict) -> int:
    num_layers = 0
    while True:
        if f"encoders_{num_layers}" not in params["params"]:
            break
        num_layers += 1
    return num_layers


def custom_normalize(tensor: jnp.ndarray, axis=0, default_constant=0.5) -> jnp.ndarray:
    """
    axis is the dimension on which to normalize
    default_constant is the value to use when the sum is zero
    """
    # Compute the sum along axis=axis and keepdims=True to maintain the dimensions for broadcasting
    # sum_tensor = jnp.sum(tensor, axis=axis, keepdims=True)

    sum_tensor = jnp.logsumexp(tensor, axis=axis, keepdims=True)

    # Get the shape of the tensor
    shape = tensor.shape
    # Get the length on axis
    length = shape[axis]

    # Create a mask where the sum is minus infinity
    mask = jnp.isinf(-sum_tensor)
    # And another mask where the sum is infinite
    inf_mask = jnp.isinf(sum_tensor)

    # jax print debugger:
    # jax.debug.print('inf_mask: {}', inf_mask)

    # Replace zero sums with ones to avoid division by zero and then divide
    result = tensor - jnp.where(mask, jnp.ones_like(sum_tensor), sum_tensor)

    # Where the sum was -infinity, replace with the constant -length where length is the length of the axis
    result = jnp.where(mask, jnp.full_like(result, fill_value=-length), result)

    return result
