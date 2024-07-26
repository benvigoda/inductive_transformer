from enum import Enum
import matplotlib.pyplot as plt


class SampleStatus(Enum):
    IN_SAMPLE = 0
    OUT_OF_SAMPLE = 1
    INVALID = 2


sample_status_names = {
    SampleStatus.IN_SAMPLE: "in sample",
    SampleStatus.OUT_OF_SAMPLE: "out of sample",
    SampleStatus.INVALID: "invalid",
}


def generate_histogram_data(sentences, classify_sentence):
    # Count the number of times each sentence appears.
    sentence_counts = {}
    for sentence in sentences:
        if sentence not in sentence_counts:
            sentence_counts[sentence] = 0
        sentence_counts[sentence] += 1

    # Classify each sentence.
    sentence_categories = {}
    for sentence in sentence_counts:
        category = classify_sentence(sentence)
        sentence_categories[sentence] = category

    # Sort the sentences by category, and then alphabetically.
    sentence_list = list(sentence_counts.keys())
    sentence_list.sort(key=lambda x: (sentence_categories[x].value, x))

    return [
        (sentence, sentence_counts[sentence], sentence_categories[sentence])
        for sentence in sentence_list
    ]


def print_histogram(data):
    for sentence, count, category in data:
        print(
            sentence,
            count,
            sample_status_names[category],
        )
