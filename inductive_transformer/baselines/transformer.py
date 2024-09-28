from collections.abc import Callable
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def scaled_dot_product(q, k, v, mask=None):
    n_elements = q.shape[-2]  # the number of elements for which we have queries/keys
    k_dim = q.shape[-1]  # the dimension of the query and key vectors
    v_dim = v.shape[-1]  # the dimension of the value vectors
    assert q.shape[-2:] == (n_elements, k_dim)
    assert k.shape[-2:] == (n_elements, k_dim)
    assert v.shape[-2:] == (n_elements, v_dim)

    # Compute the dot products between each query vector and all key vectors.
    dots = jnp.einsum("...ij,...kj->...ik", q, k)
    assert dots.shape[-2:] == (n_elements, n_elements)

    # Scale the dot products by the square root of the dimension of the key vectors.
    dots = dots / jnp.sqrt(k_dim)

    # Apply the mask (if any).
    if mask is not None:
        dots = jnp.where(mask == 0, -1e12, dots)

    # Compute the attention matrix.
    attentions = jax.nn.softmax(dots, axis=-1)
    assert attentions.shape[-2:] == (n_elements, n_elements)

    # Compute the weighted average of the values.
    values = jnp.einsum("...ij,...jk->...ik", attentions, v)
    assert values.shape[-2:] == (n_elements, v_dim)

    return values, attentions


class MultiHeadAttention(nn.Module):
    out_dim: int  # output dimension
    n_heads: int  # number of parallel heads (h)
    k_dim: int  # dimension of the query and key vectors
    v_dim: int  # dimension of the value vectors
    weight_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x, mask=None):
        n_elements, in_dim = x.shape[-2:]

        # Register parameters.
        q_weights = self.param(
            "q_weights", self.weight_init, (self.n_heads, self.k_dim, in_dim)
        )
        k_weights = self.param(
            "k_weights", self.weight_init, (self.n_heads, self.k_dim, in_dim)
        )
        v_weights = self.param(
            "v_weights", self.weight_init, (self.n_heads, self.v_dim, in_dim)
        )
        o_weights = self.param(
            "o_weights", self.weight_init, (self.out_dim, self.n_heads, self.v_dim)
        )

        # Compute the queries, keys, and values for each head.
        queries = jnp.einsum("ijk,...lk->...ilj", q_weights, x)
        keys = jnp.einsum("ijk,...lk->...ilj", k_weights, x)
        values = jnp.einsum("ijk,...lk->...ilj", v_weights, x)
        assert queries.shape[-3:] == (self.n_heads, n_elements, self.k_dim)
        assert keys.shape[-3:] == (self.n_heads, n_elements, self.k_dim)
        assert values.shape[-3:] == (self.n_heads, n_elements, self.v_dim)

        # Evaluate the attention function for each head.
        summed_values, attentions = scaled_dot_product(queries, keys, values, mask)
        assert summed_values.shape[-3:] == (self.n_heads, n_elements, self.v_dim)
        assert attentions.shape[-3:] == (self.n_heads, n_elements, n_elements)

        # Combine the outputs of the heads into a single output vector.
        outputs = jnp.einsum("ijk,...jlk->...li", o_weights, summed_values)
        assert outputs.shape[-2:] == (n_elements, self.out_dim)

        return outputs, attentions


class EncoderBlock(nn.Module):
    embedding_dim: int
    feedforward_dim: int
    n_heads: int
    k_dim: int
    v_dim: int
    dropout_rate: float
    weight_init: Callable = nn.initializers.xavier_uniform()

    def setup(self):
        # attention block
        self.self_attention = MultiHeadAttention(
            out_dim=self.embedding_dim,
            n_heads=self.n_heads,
            k_dim=self.k_dim,
            v_dim=self.v_dim,
            weight_init=self.weight_init,
        )
        self.dropout_1 = nn.Dropout(self.dropout_rate)
        self.norm_1 = nn.LayerNorm()

        # feedforward block
        self.dense_1 = nn.Dense(self.feedforward_dim)
        self.dropout_2 = nn.Dropout(self.dropout_rate)
        self.relu = nn.relu
        self.dense_2 = nn.Dense(self.embedding_dim)
        self.dropout_3 = nn.Dropout(self.dropout_rate)
        self.norm_2 = nn.LayerNorm()

    def __call__(self, x, mask=None, training=False):
        # attention block
        attention_out, _ = self.self_attention(x, mask=mask)
        attention_out = self.dropout_1(attention_out, deterministic=not training)
        x = x + attention_out
        x = self.norm_1(x)

        # feedforward block
        feedforward_out = self.dense_1(x)
        feedforward_out = self.dropout_2(feedforward_out, deterministic=not training)
        feedforward_out = self.relu(feedforward_out)
        feedforward_out = self.dense_2(feedforward_out)
        feedforward_out = self.dropout_3(feedforward_out, deterministic=not training)
        x = x + feedforward_out
        x = self.norm_2(x)
        return x


class TransformerEncoder(nn.Module):
    n_blocks: int
    embedding_dim: int
    feedforward_dim: int
    n_heads: int
    k_dim: int
    v_dim: int
    dropout_rate: float
    weight_init: Callable = nn.initializers.xavier_uniform()

    def setup(self):
        self.blocks = [
            EncoderBlock(
                embedding_dim=self.embedding_dim,
                feedforward_dim=self.feedforward_dim,
                n_heads=self.n_heads,
                k_dim=self.k_dim,
                v_dim=self.v_dim,
                dropout_rate=self.dropout_rate,
                weight_init=self.weight_init,
            )
            for _ in range(self.n_blocks)
        ]

    def __call__(self, x, mask=None, training=False):
        for block in self.blocks:
            x = block(x, mask=mask, training=training)
        return x


class PositionalEncoding(nn.Module):
    max_sequence_length: int
    embedding_dim: int

    def setup(self):
        # Pre-compute the positional encodings.
        # The numerators are just the sequence indices.
        numerator = np.arange(0, self.max_sequence_length, dtype=np.float32)[
            :, np.newaxis
        ]
        # The denominators are 10000^(i / embedding_dim) for even i, and 10000^((i - 1) /
        # embedding_dim) for odd i. We compute exp(log(1/10000^(i / embedding_dim))).
        tmp = -np.log(10000) / self.embedding_dim
        inverse_denominator = np.exp(
            tmp * np.arange(0, self.embedding_dim, 2, dtype=np.float32)
        )
        phase = numerator * inverse_denominator

        pe = np.zeros((self.max_sequence_length, self.embedding_dim))
        pe[:, 0::2] = np.sin(phase)
        pe[:, 1::2] = np.cos(phase)
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        assert x.shape[-2] <= self.max_sequence_length
        assert x.shape[-1] == self.embedding_dim
        return x + self.pe[: x.shape[-2], :]


class TransformerClassifier(nn.Module):
    sequence_length: int
    embedding_dim: int
    n_classes: int
    feedforward_dim: int
    n_blocks: int
    n_heads: int
    k_dim: int
    v_dim: int
    dropout_rate: float
    weight_init: Callable = nn.initializers.xavier_uniform()

    def setup(self):
        self.input_layer = nn.Dense(self.embedding_dim)
        self.positional_encoding = PositionalEncoding(
            max_sequence_length=self.sequence_length,
            embedding_dim=self.embedding_dim,
        )
        self.transforer_encoder = TransformerEncoder(
            n_blocks=self.n_blocks,
            embedding_dim=self.embedding_dim,
            feedforward_dim=self.feedforward_dim,
            n_heads=self.n_heads,
            k_dim=self.k_dim,
            v_dim=self.v_dim,
            dropout_rate=self.dropout_rate,
            weight_init=self.weight_init,
        )
        self.output_net = [
            nn.Dense(self.embedding_dim),
            nn.LayerNorm(),
            nn.Dropout(self.dropout_rate),
            nn.relu,
            nn.Dense(self.n_classes),
        ]

    def __call__(self, x, mask=None, training=False):
        x = self.input_layer(x)
        x = self.positional_encoding(x)
        x = self.transforer_encoder(x, mask=mask, training=training)
        for layer in self.output_net:
            x = (
                layer(x)
                if not isinstance(layer, nn.Dropout)
                else layer(x, deterministic=not training)
            )
        return x


if __name__ == "__main__":
    main_key = jax.random.PRNGKey(42)

    batch_size = 2
    sequence_length = 10
    embedding_dim = 128
    n_heads = 1
    k_dim = embedding_dim // n_heads
    v_dim = embedding_dim // n_heads
    n_classes = 5
    n_blocks = 1

    main_key, key = jax.random.split(main_key)
    in_data = jax.random.normal(key, (batch_size, sequence_length, embedding_dim))

    transformer_classifier = TransformerClassifier(
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        n_classes=n_classes,
        feedforward_dim=4 * embedding_dim,
        n_blocks=n_blocks,
        n_heads=n_heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dropout_rate=0.1,
    )

    main_key, key = jax.random.split(main_key)
    params = transformer_classifier.init(key, in_data, training=False)

    main_key, key = jax.random.split(main_key)
    output = transformer_classifier.apply(
        params, in_data, training=True, rngs={"dropout": key}
    )

    print("input\n", in_data.shape, "\n", in_data)
    print("output\n", output.shape, "\n", output)
