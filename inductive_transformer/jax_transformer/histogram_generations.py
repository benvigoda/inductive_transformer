import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


# The validation function
def validate_sentences(sentences_list):

    valid_first_pairs = {
        (1, 2): {
            ("Small", "dog"), ("Small", "canine"),
            ("Little", "dog"), ("Little", "canine"),
            ("Tiny", "dog"), ("Tiny", "canine"),
            ("Micro", "dog"), ("Micro", "canine"),
            ("Mini", "dog"), ("Mini", "canine"),
        }
    }

    valid_middle_pairs = {
        (3, 4): {
            ("often", "fears"), ("often", "avoids"),
            ("usually", "fears"), ("usually", "avoids"),
            ("commonly", "fears"), ("commonly", "avoids"),
            ("frequently", "fears"), ("frequently", "avoids"),
            ("sometimes", "chases"), ("sometimes", "intimidates"), ("sometimes", "eats"),
            ("occasionally", "chases"), ("occasionally", "intimidates"), ("occasionally", "eats"),
            ("rarely", "fears"), ("rarely", "avoids"),
            ("never", "fears"), ("never", "avoids")
        }
    }

    valid_last_pairs = {
        (5, 6): {
            ("large", "cat"), ("large", "feline"),
            ("giant", "cat"), ("giant", "feline"),
            ("huge", "cat"), ("huge", "feline"),
            ("humongous", "cat"), ("humongous", "feline"),
            ("enormous", "cat"), ("enormous", "feline"),
            ("big", "cat"), ("big", "feline")
        }
    }
    results = {}
    for sentence in sentences_list:
        words = sentence.split()

        results[sentence] = "valid"

        if len(words) != 6:
            results[sentence] = "invalid"
            continue
        
        if (words[0], words[1]) not in valid_first_pairs.get((1, 2), []):
            results[sentence] = "invalid"
            continue
        
        if (words[2], words[3]) not in valid_middle_pairs.get((3, 4), []):
            results[sentence] = "invalid"
            continue
        
        if (words[4], words[5]) not in valid_last_pairs.get((5, 6), []):
            results[sentence] = "invalid"

    return results


# Function to plot side-by-side horizontal histograms with shared y-axis
def plot_side_by_side_histograms(data1, data2):
    # Set up the figure with two subplots, sharing the y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot training data on the first subplot
    sns.barplot(x='Count', y='Sentence', data=data1, color='blue', ax=ax1)
    ax1.set_title("Training Data")
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Sentences')
    palette = {'valid': 'green', 'invalid': 'red'}
    # Plot generated data on the second subplot
    # sns.barplot(x='Count', y='Sentence', data=data2, color=data2["Color"], ax=ax2)
    sns.barplot(x='Count', y='Sentence', data=data2, hue='Status', palette=palette, ax=ax2)
    ax2.set_title("Generated Data")
    ax2.set_xlabel('Count')
    ax2.set_ylabel('')  # No y-label for the right plot

    # Adjust subplot parameters to give more space and align them nicely
    plt.subplots_adjust(wspace=0.2)  # Increase space between the plots

    # Show the plot
    plt.show()


# Function to prepare data and plot results
def histogram_results(training_sentences, generated_sentences):
    training_counts = Counter(training_sentences)
    generated_counts = Counter(generated_sentences)
    training_data = pd.DataFrame(list(training_counts.items()), columns=['Sentence', 'Count'])
    generated_data = pd.DataFrame(list(generated_counts.items()), columns=['Sentence', 'Count'])
    valid_generated = validate_sentences(training_sentences + generated_sentences)
    generated_data["Status"] = ['valid' if valid_generated[sentence] == 'valid' else 'invalid' for sentence in generated_data["Sentence"]]

    # Plot side-by-side histograms for both datasets
    plot_side_by_side_histograms(training_data, generated_data)


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


if __name__ == '__main__':
    main()
