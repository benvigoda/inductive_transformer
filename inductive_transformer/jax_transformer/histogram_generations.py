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

    palette = {"valid": "green", "invalid": "red"}
    training_palette = {"valid": "blue", "invalid": "red"}
    from matplotlib.ticker import MaxNLocator
    big_plot = True
    zoomed_plot = True
    second_plot = True
    if big_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        # Plot training data on the first subplot
        sns.barplot(x="Count", y="Sentence", data=data1, hue="Status", palette=training_palette, ax=ax1, width=bar_width)
        ax1.set_title("Training Data")
        ax1.set_xlabel("Count")
        ax1.set_ylabel("Sentences")
        ax1.legend(loc="lower left")
        # Plot generated data on the second subplot
        sns.barplot(
            x="Count", y="Sentence", data=data2, hue="Status", palette=palette, ax=ax2, width=bar_width
        )
        ax2.set_title("Generated Data")
        ax2.set_xlabel("Count")
        ax2.set_ylabel("")  # No y-label for the right plot
        ax2.legend(loc="lower right")

        # Adjust subplot parameters to give more space and align them nicely
        plt.subplots_adjust(
            left=0.18,  # Adds more space to the left
            right=0.98,  # Adds more space to the right
            wspace=0.05,  # Increase space between the plots
        )
        plt.setp(ax1.get_yticklabels(), fontsize=5)
        # Make sure the y-ticks are evenly spaced
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=50, prune='both'))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=50, prune='both'))

        # if subtitle:
        #     plt.suptitle(subtitle, fontsize=16)

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

    # Make a zoomed-in plot with only the first 100 sentences
    # Still show both histograms, but only the first 100 sentences
    if zoomed_plot:
        num_zoom_sentences = 50
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        sns.barplot(x="Count", y="Sentence", data=data1.head(num_zoom_sentences), hue="Status", palette=training_palette, ax=ax1, width=bar_width)
        ax1.set_title("Training Data")
        ax1.set_xlabel("Count")
        ax1.set_ylabel("Sentences")
        ax1.legend(loc="lower left")
        # Select data from the second dataset that corresponds to the first num_zoom_sentences sentences in the first dataset
        tmp_data2 = data2[data2["Sentence"].isin(data1["Sentence"].head(num_zoom_sentences))]
        # Add sentences from data2 to have num_zoom_sentences sentences in total
        data2 = pd.concat([tmp_data2, data2]).drop_duplicates()
        sns.barplot(
            x="Count", y="Sentence", data=data2.head(num_zoom_sentences), hue="Status", palette=palette, ax=ax2, width=bar_width
        )
        ax2.set_title("Generated Data")
        ax2.set_xlabel("Count")
        ax2.set_ylabel("")  # No y-label for the right plot
        ax2.legend(loc="lower right")
        plt.subplots_adjust(left=0.18, right=0.98, wspace=0.05)
        plt.setp(ax1.get_yticklabels(), fontsize=5)
        # Actually, only show every 10th y-tick label
        # for i, label in enumerate(ax1.yaxis.get_ticklabels()):
        #     if i % 4 != 0:
        #         label.set_visible(False)
        #     else:
        #         label.set_visible(True)
        # Make sure the y-ticks are evenly spaced
        # ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        # ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        # Save the plot
        if plot_file_name:
            if folder:
                file_path = os.path.join(folder, plot_file_name[:-4] + "_zoom.png")
                plt.savefig(file_path)
            else:
                plt.savefig(plot_file_name[:-4] + "_zoom.png")
            plt.close()

    # Make a another plot with the same original data, but only show the second histogram
    # This is useful for saving the second histogram separately
    if second_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            x="Count", y="Sentence", data=data2, hue="Status", palette=palette, ax=ax, width=bar_width
        )
        ax.set_title("Generated Data")
        ax.set_xlabel("Count")
        ax.set_ylabel("")  # No y-label for the right plot
        ax.legend(loc="lower right")
        plt.subplots_adjust(left=0.4, right=0.98, wspace=0.05)
        plt.setp(ax.get_yticklabels(), fontsize=5)
        # Actually, only show every 10th y-tick label
        # for i, label in enumerate(ax.yaxis.get_ticklabels()):
        #     if i % 5 != 0:
        #         label.set_visible(False)
        #     else:
        #         label.set_visible(True)
        # Make sure the y-ticks are evenly spaced
        # ax.yaxis.set_major_locator(MaxNLocator(nbins=50, prune='both'))
        # Save the plot
        if plot_file_name:
            if folder:
                file_path = os.path.join(folder, plot_file_name[:-4] + "_second.png")
                plt.savefig(file_path)
            else:
                plt.savefig(plot_file_name[:-4] + "_second.png")
            plt.close()


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
    training_data["Status"] = [
        "valid" if valid_generated[sentence] == "valid" else "invalid"
        for sentence in training_data["Sentence"]
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
