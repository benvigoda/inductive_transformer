import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


# The validation function
def validate_sentences(sentences_list, num_words=6):

    if num_words == 6:
        valid_first_pairs = {
            (1, 2): {
                ("Small", "dogs"), ("Small", "canines"),
                ("Little", "dogs"), ("Little", "canines"),
                ("Tiny", "dogs"), ("Tiny", "canines"),
                ("Micro", "dogs"), ("Micro", "canines"),
                ("Mini", "dogs"), ("Mini", "canines"),
            }
        }
    else:
        valid_first_pairs = {
            (1, 2): {
                ("Small", "dogs"), ("Small", "canines"),
                ("Little", "dogs"), ("Little", "canines"),
                ("Tiny", "dogs"), ("Tiny", "canines"),
                ("Micro", "dogs"), ("Micro", "canines"),
                ("Mini", "dogs"), ("Mini", "canines"),
                ("Extralarge", "cats"), ("Extralarge", "felines"),
                ("Gargantuan", "cats"), ("Gargantuan", "felines"),
                ("Large", "cats"), ("Large", "felines"),
                ("Giant", "cats"), ("Giant", "felines"),
                ("Huge", "cats"), ("Huge", "felines"),
                ("Humongous", "cats"), ("Humongous", "felines"),
                ("Enormous", "cats"), ("Enormous", "felines"),
                ("Big", "cats"), ("Big", "felines"),
                ("Pico", "dogs"), ("Pico", "canines"),
                ("Femto", "dogs"), ("Femto", "canines"),
                ("Diminimus", "dogs"), ("Diminimus", "canines"),
                ("Itty", "dogs"), ("Itty", "canines"),
                ("Teenyweeny", "dogs"), ("Teenyweeny", "canines"),
            }
        }

    valid_middle_pairs = {
        (3, 4): {
            ("often", "fear"), ("often", "avoid"),
            ("usually", "fear"), ("usually", "avoid"),
            ("commonly", "fear"), ("commonly", "avoid"),
            ("frequently", "fear"), ("frequently", "avoid"),
            ("sometimes", "chase"), ("sometimes", "intimidate"), ("sometimes", "eat"),
            ("occasionally", "chase"), ("occasionally", "intimidate"), ("occasionally", "eat"),
            ("rarely", "fear"), ("rarely", "avoid"),
            ("never", "fear"), ("never", "avoid")
        }
    }

    valid_last_pairs = {
        (5, 6): {
            ("large", "cats"), ("large", "felines"),
            ("giant", "cats"), ("giant", "felines"),
            ("huge", "cats"), ("huge", "felines"),
            ("humongous", "cats"), ("humongous", "felines"),
            ("enormous", "cats"), ("enormous", "felines"),
            ("big", "cats"), ("big", "felines")
        }
    }

    results = {}
    for sentence in sentences_list:
        words = sentence.split()
        results[sentence] = "valid"

        if len(words) != num_words:
            results[sentence] = "invalid"
            continue
        
        if num_words >= 2 and (words[0], words[1]) not in valid_first_pairs.get((1, 2), []):
            results[sentence] = "invalid"
            continue
        
        if num_words >= 4 and (words[2], words[3]) not in valid_middle_pairs.get((3, 4), []):
            results[sentence] = "invalid"
            continue
        
        if num_words >= 6 and (words[4], words[5]) not in valid_last_pairs.get((5, 6), []):
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

    plt.setp(ax1.get_yticklabels(), fontsize=5)

    # Show the plot
    plt.show()


# Function to prepare data and plot results
def histogram_results(training_sentences, generated_sentences):
    num_words = len(training_sentences[0].split())
    training_counts = Counter(training_sentences)
    generated_counts = Counter(generated_sentences)
    training_data = pd.DataFrame(list(training_counts.items()), columns=['Sentence', 'Count'])
    generated_data = pd.DataFrame(list(generated_counts.items()), columns=['Sentence', 'Count'])
    valid_generated = validate_sentences(training_sentences + generated_sentences, num_words=num_words)
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
