import flax.linen as nn


class FullyConnectedSeqToSeq(nn.Module):
    # Each entry gives the number of units in a dense hidden layer.
    layers: list[int]

    @nn.compact
    def __call__(self, x):
        """
        Maps a tensor of shape (batch size, sentence length, vocab size) to a tensor of the same
        shape, representing logits over tokens for each position (and each sample).
        """

        assert x.ndim == 3
        batch_size, sentence_length, vocab_size = x.shape

        x = x.reshape(batch_size, sentence_length * vocab_size)
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = nn.tanh(x)
        x = nn.Dense(sentence_length * vocab_size)(x)
        x = x.reshape(batch_size, sentence_length, vocab_size)
        return x


class FullyConnectedAutoregressive(nn.Module):
    # Each entry gives the number of units in a dense hidden layer.
    layers: list[int]

    @nn.compact
    def __call__(self, x):
        """
        Maps a tensor of shape (batch size, sentence length, vocab size) to a tensor of shape (batch
        size, vocab size), representing logits over tokens for the next position. The expectation is
        that all tokens at the next position and later will be replaced with BLANK.
        """

        assert x.ndim == 3
        batch_size, sentence_length, vocab_size = x.shape

        x = x.reshape(batch_size, sentence_length * vocab_size)
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = nn.tanh(x)
        x = nn.Dense(vocab_size)(x)
        x = x.reshape(batch_size, vocab_size)
        return x
