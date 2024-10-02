from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


class Dataset(NamedTuple):
    word_to_id: dict[str, int]
    id_to_word: dict[int, str]
    data: jax.Array  # axes are sentence, word position (in sentence)
    includes_blank_token: bool = True

    @property
    def n_sentences(self):
        return self.data.shape[0]

    @property
    def sentence_length(self):
        return self.data.shape[1]

    @property
    def vocab_size(self):
        return len(self.word_to_id)

    @property
    def blank_token(self):
        if self.includes_blank_token:
            return self.vocab_size - 1
        else:
            raise ValueError("No blank token in dataset.")

    def ids_to_strings(self, ids):
        ids = np.asarray(ids).tolist()
        return [" ".join([self.id_to_word[id] for id in sentence]) for sentence in ids]

    def strings_to_ids(self, strings):
        return jnp.array(
            [
                [self.word_to_id[word] for word in sentence.split()]
                for sentence in strings
            ]
        )


def load_dataset(filepath) -> Dataset:
    """Loads data from a file where each line is a sentence."""

    # Load the file into memory.
    with open(filepath, "r") as f:
        # Split on '\n' and drop empty
        sentences = [x.strip() for x in f]
        sentences = [x for x in sentences if x != ""]
        n_sentences = len(sentences)
        assert n_sentences > 0

    # Split on ' '.
    words = [x.split() for x in sentences]
    return make_dataset_from_sentences(words)


def load_dataset_old(filepath) -> Dataset:
    """Loads sentences separated by '.' from a file and returns a Dataset."""

    # Load the file into memory.
    with open(filepath, "r") as f:
        data = f.read()

    # Split on '.' and drop empty sentences.
    sentences = [x.strip() for x in data.strip().split(".")]
    sentences = [x for x in sentences if x != ""]
    n_sentences = len(sentences)
    assert n_sentences > 0

    # Split on ' '.
    words = [x.split() for x in sentences]
    return make_dataset_from_sentences(words)


def make_dataset_from_sentences(words, include_blank_token=True) -> Dataset:
    # Words should be a list of lists of strings (or more generally an iterable of iterables of
    # strings).
    n_sentences = len(words)

    # Check that all sentences have the same length.
    lengths = set([len(x) for x in words])
    assert len(lengths) == 1
    sentence_length = list(lengths)[0]

    # Create a vocabulary.
    word_to_id = {}
    for sentence in words:
        for word in sentence:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
    if include_blank_token:
        word_to_id["BLANK"] = len(word_to_id)
    id_to_word = {v: k for k, v in word_to_id.items()}

    # print("Vocabulary")
    # vocab_size = len(word_to_id)
    # for id in range(vocab_size):
    #     print(id, id_to_word[id])
    # print("")

    # Convert words to ids.
    word_ids = [[word_to_id[word] for word in sentence] for sentence in words]
    word_ids = jnp.array(word_ids)
    assert word_ids.shape == (n_sentences, sentence_length)

    return Dataset(word_to_id, id_to_word, word_ids, include_blank_token)
