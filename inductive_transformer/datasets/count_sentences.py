import click
from collections import Counter


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
def count_sentences(input_file):
    """Counts how many times each sentence occurs in the input file."""
    with open(input_file, "r") as infile:
        sentences = infile.readlines()

    # Strip newlines and count occurrences of each sentence
    sentences = [sentence.strip() for sentence in sentences]
    sentence_counts = Counter(sentences)

    # Print the number of unique sentences
    unique_sentences = len(sentence_counts)
    print(f"Number of unique sentences: {unique_sentences}")

    # Count how many sentences appear once, twice, etc.
    count_occurrences = Counter(sentence_counts.values())

    # Print the number of sentences for each occurrence count
    for occurrence, count in sorted(count_occurrences.items()):
        print(f"Sentences that appear {occurrence} time(s): {count}")


if __name__ == "__main__":
    count_sentences()
