import flax.linen as nn


class FullyConnectedSeqToSeq(nn.Module):
    # Each entry gives the number of units in a dense hidden layer.
    layers: list[int]

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 3
        batch_size, sentence_length, vocab_size = x.shape

        x = x.reshape(batch_size, sentence_length * vocab_size)
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = nn.tanh(x)
        x = nn.Dense(sentence_length * vocab_size)(x)
        x = x.reshape(batch_size, sentence_length, vocab_size)
        return x
