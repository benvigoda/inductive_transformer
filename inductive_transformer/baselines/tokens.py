import numpy as np


def load_dataset(filepath):
    with open(filepath, 'r') as f:
        data = f.read()

    # Split on '.' and drop empty sentences.
    sentences = [x.strip() for x in data.strip().split('.')]
    sentences = [x for x in sentences if x != '']
    assert len(sentences) > 0

    # Split on ' '.
    words = [x.split() for x in sentences]

    # Check that all sentences have the same length.
    lengths = set([len(x) for x in words])
    assert len(lengths) == 1

    # Create a vocabulary.
    word_to_id = {}
    for sentence in words:
        for word in sentence:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
    word_to_id["BLANK"] = len(word_to_id)
    id_to_word = {v: k for k, v in word_to_id.items()}
    vocab_size = len(word_to_id)

    print("Vocabulary")
    for id in range(vocab_size):
        print(id, id_to_word[id])
    print("")


if __name__ == "__main__":
    load_dataset("../../32_2_layer_sentences.txt")
