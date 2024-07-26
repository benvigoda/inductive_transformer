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


def plot_histogram(data, filename):
    # Extract the data for plotting
    sentences = [item[0] for item in data]
    counts = [item[1] for item in data]
    categories = [item[2] for item in data]

    # Reverse the order of the data so that the first item is at the top
    sentences = sentences[::-1]
    counts = counts[::-1]
    categories = categories[::-1]

    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Create a color map for the categories
    color_map = {
        SampleStatus.IN_SAMPLE: "blue",
        SampleStatus.OUT_OF_SAMPLE: "green",
        SampleStatus.INVALID: "red",
    }

    # Plot the horizontal bar chart
    ax.barh(sentences, counts, color=[color_map[category] for category in categories])

    # Set labels and title
    ax.set_xlabel("Count")
    ax.set_ylabel("Sentence")
    ax.set_title("Histogram")

    # Ensure there is enough space for the labels
    fig.tight_layout()

    # Show the plot
    plt.savefig(filename)
