import click


@click.command()
@click.option("--input_file", "-i", type=click.Path(exists=True))
@click.option("--output_file", "-o", type=click.Path(writable=True))
def split_sentences(input_file, output_file):
    with open(input_file, "r") as infile:
        text = infile.read()

    # Split sentences by period followed by a space
    sentences = text.split(". ")

    # Remove any trailing period and add a newline for each sentence
    cleaned_sentences = [
        sentence.strip().rstrip(".") for sentence in sentences if sentence != ""
    ]

    # Write the cleaned sentences to the output file
    with open(output_file, "w") as outfile:
        for sentence in cleaned_sentences:
            outfile.write(sentence + "\n")


if __name__ == "__main__":
    split_sentences()
