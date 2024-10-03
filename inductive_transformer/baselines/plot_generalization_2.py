import click
import numpy as np
import matplotlib.pyplot as plt

from plot_generalization import plot_generalization_performance


def load_sample_data(log_file):
    data = []
    in_epoch_block = False

    with open(log_file, "r") as file:
        for line in file:
            if line.startswith("epoch,"):
                in_epoch_block = True
                continue

            if in_epoch_block:
                # Stop if we encounter the section after the epoch block
                if not line.strip() or not line[0].isdigit():
                    break

                # Parse stats
                parts = line.split()
                assert len(parts) == 6
                epoch = int(parts[0])
                n_in_sample = int(parts[3])
                n_out_of_sample = int(parts[4])
                n_invalid = int(parts[5])
                if n_in_sample + n_out_of_sample + n_invalid == 0:
                    continue
                data.append([epoch, n_in_sample, n_out_of_sample, n_invalid])

    return np.array(data, dtype=np.int32)


def plot_generalization_performance(data):
    # Split the data into the respective categories
    epochs = data[:, 0]
    n_in_sample = data[:, 1]
    n_out_of_sample = data[:, 2]
    n_invalid = data[:, 3]

    # Create the stacked line plot
    fig, ax = plt.subplots()
    # ax.set_yscale('log')

    ax.stackplot(
        epochs,
        n_in_sample,
        n_out_of_sample,
        n_invalid,
        labels=["In Sample", "Out of Sample", "Invalid"],
        colors=["blue", "green", "red"],
    )

    # Add labels and title
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count")
    ax.set_title("Generalization Performance vs Epoch")
    ax.legend(loc="lower left")

    return fig


def main():
    data_file = "log_48_6_layer_sentences_balanced_newline.txt"
    data = load_sample_data(data_file)
    fig = plot_generalization_performance(data)
    fig.savefig("generalization_vs_epoch.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
