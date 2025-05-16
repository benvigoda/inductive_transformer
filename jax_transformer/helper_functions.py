import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax.nn import logsumexp

EPSILON = 1e-20
IMPROBABLE = -9
PROBABLE = 0 - 1e-9


def get_num_layers(params: dict) -> int:
    num_layers = 0
    while True:
        if f"encoders_{num_layers}" not in params["params"]:
            break
        num_layers += 1
    return num_layers


# when missing both bound_weights and bound_activations we hit nans with this command:
# PYTHONPATH=. python jax_transformer/train.py 48_6_layer_sentences_balanced_dogs_birds_all_synonyms.txt --prompt_text inference_text.txt 
# --num_layer 6 --layer_width 2 --num_samples 10 --num_epochs 100 --silence_print --seed 2768615008 --initialize_weights

# when we bound_weights but not activations we still hit nans

# when we bound_activations, but not weights we are good - no nans

# if we bound both, obviously we are bounding activations so again we're good.

# we still need masking 



def bound_weights(tensor: jnp.ndarray, upper_bound = 0.0, lower_bound= -1000.0):
    return tensor
    # nan (ArrayLike) – value to substitute for NaN entries. FIXME: Should it be lower_bound or upper_bound?
    jax.numpy.nan_to_num(tensor, nan=lower_bound, posinf=upper_bound, neginf=lower_bound)

    # https://docs.jax.dev/en/latest/_autosummary/jax.numpy.clip.html 
    return jnp.clip(tensor, min=lower_bound, max=upper_bound)

def bound_activations(tensor: jnp.ndarray, upper_bound = 0.0, lower_bound= -1000.0):
    # return tensor
    # nan (ArrayLike) – value to substitute for NaN entries. FIXME: Should it be lower_bound or upper_bound?
    jax.numpy.nan_to_num(tensor, nan=lower_bound, posinf=upper_bound, neginf=lower_bound)

    # https://docs.jax.dev/en/latest/_autosummary/jax.numpy.clip.html 
    return jnp.clip(tensor, min=lower_bound, max=upper_bound)
 


def custom_normalize(tensor: jnp.ndarray, axis=0, default_constant=0.5) -> jnp.ndarray:
    """
    axis is the dimension on which to normalize
    default_constant is the value to use when the sum is zero
    """
    # Compute the sum along axis=axis and keepdims=True to maintain the dimensions for broadcasting
    # sum_tensor = jnp.sum(tensor, axis=axis, keepdims=True)

    sum_tensor = logsumexp(tensor, axis=axis, keepdims=True)

    # Get the shape of the tensor
    shape = tensor.shape
    # Get the length on axis
    length = shape[axis]

    # Create a mask where the sum is minus infinity
    mask = jnp.isinf(-sum_tensor)

    # Replace -inf sums with zeros to avoid subtracting -infinity and then subtract
    result = tensor - jnp.where(mask, jnp.zeros_like(sum_tensor), sum_tensor)

    # Where the sum was -infinity, replace with the constant -length where length is the length of the axis
    result = jnp.where(mask, jnp.full_like(result, fill_value=-length), result)

    return result
