import jax.numpy as jnp  # type: ignore
import jax  # type: ignore
from flax import linen as nn  # type: ignore 

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


def shift_up_to_make_all_elements_positive(x, axis=0):
    # min(x) is positive, add 0
    # min(x) is negative, add -min(x)
    # but we do not want to use an IF because jax hates that
    # but relu(-min(x)) is zero when min(x) is positive 
    # and -min(x) when min(x) is negative
    # which is what we want!
    
    return nn.relu(x) + EPSILON
    # return x + nn.relu(-jnp.min(x, axis=axis, keepdims=True)) + EPSILON


def custom_normalize(x, axis):

    x += EPSILON

    # Compute the norm along the specified axis, adding EPSILON to prevent division by zero
    norm = jnp.sum(x, axis=axis, keepdims=True)
    # Normalize the input by dividing by the norm
    
    return (x / norm)


# OLD VERSION
# def custom_normalize_old(tensor: jnp.ndarray, axis=0, default_constant=0.5) -> jnp.ndarray:
#     """
#     axis is the dimension on which to normalize
#     default_constant is the value to use when the sum is zero
#     """
#     # Compute the sum along axis=axis and keepdims=True to maintain the dimensions for broadcasting
#     sum_tensor = jnp.sum(tensor, axis=axis, keepdims=True)

#     # Get the shape of the tensor
#     shape = tensor.shape
#     # Get the length on axis
#     length = shape[axis]

#     # Create a mask where the sum is zero
#     mask = sum_tensor == 0
#     # And another mask where the sum is infinite
#     inf_mask = jnp.isinf(sum_tensor)

#     # jax print debugger:
#     # jax.debug.print('inf_mask: {}', inf_mask)


#     # Replace zero sums with ones to avoid division by zero and then divide
#     result = tensor / jnp.where(mask, jnp.ones_like(sum_tensor), sum_tensor)

#     # Where the sum was zero, replace with the constant 1/length where length is the length of the axis
#     result = jnp.where(mask, jnp.full_like(result, fill_value=1/length), result)

#     return result
