from flax import linen as nn  # type: ignore
import jax.numpy as jnp  # type: ignore
from typing import Callable
from inductive_transformer.jax_transformer.helper_functions import (
    custom_normalize,
    EPSILON,
)
import jax  # type: ignore


class EncoderAttentionPi(nn.Module):
    vocab_size: int
    layer_width: int
    weight_init: Callable = nn.initializers.uniform(scale=1.0, dtype=jnp.float32)

    @nn.compact
    def __call__(self, v):
        assert v.shape == (self.layer_width, self.layer_width)
        # we expect v to be already normalized categorical
        weights = self.param(
            "weights", self.weight_init, (self.layer_width, self.layer_width)
        )
        weights_nan = jnp.isnan(weights).any()
        try:
            if weights_nan.val[0]:
                print("nan in weights at encoder_attention_pi")
        except:
            if weights_nan:
                print("nan in weights at encoder_attention_pi")
        prob_weights = nn.relu(weights) + EPSILON

        prob_weights = custom_normalize(prob_weights, axis=1)

        # in the future we may want to normalize v here for good measure
        v = custom_normalize(v, axis=1)

        # element-wise product of weight vector and token vector for each column in the layer
        y = prob_weights * v

        # make it an inner product by taking a sum along the choice dimension
        y = jnp.sum(
            y, axis=0, keepdims=True
        )  # after summing it is size = (1, layer_width)
        assert y.shape == (1, self.layer_width)
        y_nan = jnp.isnan(y).any()
        try:
            if y_nan.val[0]:
                print("nan in y at encoder_attention_pi")
        except:
            if y_nan:
                print("nan in y at encoder_attention_pi")
        # we had to remove this since otherwise, y_categorical would have 0.5's instead of 1's,
        # when it is certain on both values of the layer_width index:
        # y = custom_normalize(y, axis=1)

        return y  # y is categorical
