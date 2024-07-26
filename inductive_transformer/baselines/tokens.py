from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


class Dataset(NamedTuple):
    word_to_id: dict[str, int]
    id_to_word: dict[int, str]
    data: jax.Array  # axes are sentence, word position (in sentence), word id

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
        return self.vocab_size - 1

    def ids_to_strings(self, ids):
        ids = np.asarray(ids).tolist()
        return [" ".join([self.id_to_word[id] for id in sentence]) for sentence in ids]


def load_dataset(filepath):
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


def make_dataset_from_sentences(words):
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

    return Dataset(word_to_id, id_to_word, word_ids)
