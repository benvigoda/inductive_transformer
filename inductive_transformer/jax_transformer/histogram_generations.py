import os
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from collections import Counter


# The validation function
def validate_sentences(sentences_list, grammar):
    results = {}
    for sentence in sentences_list:
        if grammar.is_valid(sentence):
            results[sentence] = "valid"
        else:
            results[sentence] = "invalid"

    return results


# Function to plot side-by-side horizontal histograms with shared y-axis
def plot_side_by_side_histograms(data1, data2, subtitle=None, plot_file_name=None, folder=None, bar_width=0.5):
    # Set up the figure with two subplots, sharing the y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Plot training data on the first subplot
    sns.barplot(x="Count", y="Sentence", data=data1, color="blue", ax=ax1, width=bar_width)
    ax1.set_title("Training Data")
    ax1.set_xlabel("Count")
    ax1.set_ylabel("Sentences")
    palette = {"valid": "green", "invalid": "red"}
    # Plot generated data on the second subplot
    sns.barplot(
        x="Count", y="Sentence", data=data2, hue="Status", palette=palette, ax=ax2, width=bar_width
    )
    ax2.set_title("Generated Data")
    ax2.set_xlabel("Count")
    ax2.set_ylabel("")  # No y-label for the right plot

    # Adjust subplot parameters to give more space and align them nicely
    plt.subplots_adjust(
        left=0.18,  # Adds more space to the left
        right=0.98,  # Adds more space to the right
        wspace=0.05,  # Increase space between the plots
    )
    plt.setp(ax1.get_yticklabels(), fontsize=5)
    # Actually, only show every 10th y-tick label
    # for i, label in enumerate(ax1.yaxis.get_ticklabels()):
    #     if i % 50 != 0:
    #         label.set_visible(False)
    #     else:
    #         label.set_visible(True)
    # Make sure the y-ticks are evenly spaced
    from matplotlib.ticker import MaxNLocator
    # ax1.yaxis.set_major_locator(plt.MaxNLocator(100))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=50, prune='both'))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=50, prune='both'))

    if subtitle:
        plt.suptitle(subtitle, fontsize=16)

    # Show the plot
    if plot_file_name:
        if folder:
            file_path = os.path.join(folder, plot_file_name)
            plt.savefig(file_path)
        else:
            plt.savefig(plot_file_name)
        plt.close()
    else:
        plt.show()


# Function to prepare data and plot results
def histogram_results(training_sentences, generated_sentences, grammar, catsanddogs=False, subtitle=None, plot_file_name=None, folder=None):
    # Count the occurrences of each sentence in both datasets
    training_counts = Counter(training_sentences)
    generated_counts = Counter(generated_sentences)
    training_data = pd.DataFrame(
        list(training_counts.items()), columns=["Sentence", "Count"]
    )
    generated_data = pd.DataFrame(
        list(generated_counts.items()), columns=["Sentence", "Count"]
    )
    valid_generated = validate_sentences(
        training_sentences + generated_sentences, grammar
    )
    generated_data["Status"] = [
        "valid" if valid_generated[sentence] == "valid" else "invalid"
        for sentence in generated_data["Sentence"]
    ]

    # Plot side-by-side histograms for both datasets
    plot_side_by_side_histograms(training_data, generated_data, subtitle=subtitle, plot_file_name=plot_file_name, folder=folder)


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
