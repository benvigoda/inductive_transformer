import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from collections import Counter
from synonyms import Synonyms  # type: ignore


# The validation function
def validate_sentences(sentences_list, num_words=6, catsanddogs=False):
    synonyms = Synonyms()
    if catsanddogs:
        synonyms.cats_and_dogs_overwrite()
    valid_pairs = synonyms.get_valid_pairs()

    results = {}
    for sentence in sentences_list:
        words = sentence.lower().split()
        results[sentence] = "valid"

        if len(words) != num_words:
            results[sentence] = "invalid"
            continue

        relevant_pairs = {
            key: value
            for key, value in valid_pairs.items()
            if all(k < num_words for k in key)
        }
        for key, value in relevant_pairs.items():
            if (words[key[0]], words[key[1]]) not in value:
                results[sentence] = "invalid"
                break

    return results


# Function to plot side-by-side horizontal histograms with shared y-axis
def plot_side_by_side_histograms(data1, data2, subtitle=None, plot_file_name=None):
    # Set up the figure with two subplots, sharing the y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Plot training data on the first subplot
    sns.barplot(x="Count", y="Sentence", data=data1, color="blue", ax=ax1)
    ax1.set_title("Training Data")
    ax1.set_xlabel("Count")
    ax1.set_ylabel("Sentences")
    palette = {"valid": "green", "invalid": "red"}
    # Plot generated data on the second subplot
    sns.barplot(
        x="Count", y="Sentence", data=data2, hue="Status", palette=palette, ax=ax2
    )
    ax2.set_title("Generated Data")
    ax2.set_xlabel("Count")
    ax2.set_ylabel("")  # No y-label for the right plot

    # Adjust subplot parameters to give more space and align them nicely
    plt.subplots_adjust(wspace=0.2)  # Increase space between the plots

    plt.setp(ax1.get_yticklabels(), fontsize=5)
    if subtitle:
        plt.suptitle(subtitle, fontsize=16)

    # Show the plot
    if plot_file_name:
        plt.savefig(plot_file_name)
    else:
        plt.show()


# Function to prepare data and plot results
def histogram_results(training_sentences, generated_sentences, catsanddogs=False, subtitle=None, plot_file_name=None):
    num_words = len(training_sentences[0].split())
    training_counts = Counter(training_sentences)
    generated_counts = Counter(generated_sentences)
    training_data = pd.DataFrame(
        list(training_counts.items()), columns=["Sentence", "Count"]
    )
    generated_data = pd.DataFrame(
        list(generated_counts.items()), columns=["Sentence", "Count"]
    )
    valid_generated = validate_sentences(
        training_sentences + generated_sentences, num_words=num_words, catsanddogs=catsanddogs
    )
    generated_data["Status"] = [
        "valid" if valid_generated[sentence] == "valid" else "invalid"
        for sentence in generated_data["Sentence"]
    ]

    # Plot side-by-side histograms for both datasets
    plot_side_by_side_histograms(training_data, generated_data, subtitle=subtitle, plot_file_name=plot_file_name)


def main():
    training_data = [
        "tiny dog often avoids large cat",  # valid
        "mini canine usually fears huge feline",  # valid
    ]
    generated_data = [
        "tiny dog often avoids large cat",  # valid
        "mini canine usually fears huge feline",  # valid
        "tiny dog often avoids large cat",  # valid
        "mini canine usually fears huge feline",  # valid
        "tiny dog often avoids dog cat",  # invalid
        "dog tiny avoids often cat cat",  # invalid
        "micro cat rarely fears big dog",  # valid
        "mini canine sometimes chases huge cat",  # valid
        "little dog occasionally intimidates enormous cat",  # valid
        "micro dog rarely fears big cat",  # valid
    ]
    histogram_results(training_data, generated_data)


if __name__ == "__main__":
    main()
