from collections.abc import Callable
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def scaled_dot_product(q, k, v, mask=None):
    """The mask should have shape (n_elements, n_elements). The first axis is the query axis, and
    the second the key axis. So mask[i, j] should be 0 if the i-th element should not attend to the
    j-th element."""

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
        # Usually both in_dim and out_dim are the embedding dimension.
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

        # Combine the outputs of the heads into a single output vector. Usually this is described as
        # concatenating the outputs of the heads and applying a linear transformation in the form of
        # a matrix. The version below is algebraically equivalent.
        outputs = jnp.einsum("ijk,...jlk->...li", o_weights, summed_values)
        assert outputs.shape[-2:] == (n_elements, self.out_dim)

        return outputs, attentions


class FeedForward(nn.Module):
    embedding_dim: int
    feedforward_dim: int
    dropout_rate: float

    def setup(self):
        self.dense_1 = nn.Dense(self.feedforward_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.relu
        self.dense_2 = nn.Dense(self.embedding_dim)

    def __call__(self, x, deterministic=True):
        x = self.dense_1(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.relu(x)
        x = self.dense_2(x)
        return x


class AddAndNorm(nn.Module):
    dropout_rate: float

    def setup(self):
        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm = nn.LayerNorm()

    def __call__(self, x, y, deterministic=True):
        return self.norm(x + self.dropout(y, deterministic=deterministic))


class EncoderLayer(nn.Module):
    embedding_dim: int
    feedforward_dim: int
    n_heads: int
    k_dim: int
    v_dim: int
    dropout_rate: float
    weight_init: Callable = nn.initializers.xavier_uniform()

    def setup(self):
        # attention block
        self.attention = MultiHeadAttention(
            out_dim=self.embedding_dim,
            n_heads=self.n_heads,
            k_dim=self.k_dim,
            v_dim=self.v_dim,
            weight_init=self.weight_init,
        )
        self.add_and_norm_1 = AddAndNorm(dropout_rate=self.dropout_rate)

        # feedforward block
        self.feedforward = FeedForward(
            embedding_dim=self.embedding_dim,
            feedforward_dim=self.feedforward_dim,
            dropout_rate=self.dropout_rate,
        )
        self.add_and_norm_2 = AddAndNorm(dropout_rate=self.dropout_rate)

    def __call__(self, x, mask=None, deterministic=True):
        # attention block
        y, _ = self.attention(x, mask=mask)
        x = self.add_and_norm_1(x, y, deterministic=deterministic)

        # feedforward block
        y = self.feedforward(x, deterministic=deterministic)
        x = self.add_and_norm_2(x, y, deterministic=deterministic)
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
        self.layers = [
            EncoderLayer(
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
        for block in self.layers:
            x = block(x, mask=mask, deterministic=not training)
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
        # embedding_dim) for odd i. We work in log space: log(1/10000^(i / embedding_dim)) =
        # -log(10000) * i / embedding_dim.
        i = np.arange(0, self.embedding_dim, 2, dtype=np.float32)
        log_inverse_denominator = -np.log(10000) * i / self.embedding_dim
        inverse_denominator = np.exp(log_inverse_denominator)
        phase = numerator * inverse_denominator

        pe = np.zeros((self.max_sequence_length, self.embedding_dim))
        pe[:, 0::2] = np.sin(phase)
        pe[:, 1::2] = np.cos(phase)
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        assert x.shape[-2] <= self.max_sequence_length
        assert x.shape[-1] == self.embedding_dim
        return x + self.pe[: x.shape[-2], :]


class Preprocessor(nn.Module):
    sequence_length: int
    n_classes: int
    embedding_dim: int
    dropout_rate: float

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.n_classes, features=self.embedding_dim
        )
        self.positional_encoding = PositionalEncoding(
            max_sequence_length=self.sequence_length,
            embedding_dim=self.embedding_dim,
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, deterministic=True):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


class TransformerClassifier(nn.Module):
    sequence_length: int
    n_classes: int
    embedding_dim: int
    feedforward_dim: int
    n_blocks: int
    n_heads: int
    k_dim: int
    v_dim: int
    dropout_rate: float
    weight_init: Callable = nn.initializers.xavier_uniform()

    def setup(self):
        self.preprocessor = Preprocessor(
            sequence_length=self.sequence_length,
            n_classes=self.n_classes,
            embedding_dim=self.embedding_dim,
            dropout_rate=self.dropout_rate,
        )
        self.encdoer = TransformerEncoder(
            n_blocks=self.n_blocks,
            embedding_dim=self.embedding_dim,
            feedforward_dim=self.feedforward_dim,
            n_heads=self.n_heads,
            k_dim=self.k_dim,
            v_dim=self.v_dim,
            dropout_rate=self.dropout_rate,
            weight_init=self.weight_init,
        )
        self.output_net = nn.Dense(self.n_classes)

    def __call__(self, x, mask=None, training=False):
        x = self.preprocessor(x)
        x = self.encdoer(x, mask=mask, training=training)
        x = self.output_net(x)
        x = nn.log_softmax(x, axis=-1)
        return x


def make_causal_attention_mask(sequence_length):
    shape = (sequence_length, sequence_length)
    return jnp.triu(jnp.ones(shape, dtype=jnp.uint8), k=1).transpose()


if __name__ == "__main__":
    main_key = jax.random.PRNGKey(42)

    batch_size = 2
    sequence_length = 10
    n_classes = 5
    embedding_dim = 128
    feedforward_dim = 4 * embedding_dim
    n_heads = 1
    k_dim = embedding_dim // n_heads
    v_dim = embedding_dim // n_heads
    n_blocks = 1
    dropout_rate = 0.1

    mask = make_causal_attention_mask(sequence_length)
    print("mask\n", mask.shape, "\n", mask)

    main_key, key = jax.random.split(main_key)
    in_data = jax.random.randint(key, (batch_size, sequence_length), 0, n_classes)

    transformer_classifier = TransformerClassifier(
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        n_classes=n_classes,
        feedforward_dim=feedforward_dim,
        n_blocks=n_blocks,
        n_heads=n_heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dropout_rate=dropout_rate,
    )

    main_key, key = jax.random.split(main_key)
    params = transformer_classifier.init(key, in_data, training=False)

    main_key, key = jax.random.split(main_key)
    output = transformer_classifier.apply(
        params, in_data, mask=mask, training=True, rngs={"dropout": key}
    )

    print("input\n", in_data.shape, "\n", in_data)
    print("output log probs\n", output.shape, "\n", output)
    probs = jnp.exp(output)
    print("output probs\n", output.shape, "\n", probs)
    prob_sums = probs.sum(axis=-1)
    print("output prob sums\n", prob_sums.shape, "\n", prob_sums)
