import numpy as np
import matplotlib.pyplot as plt


def load_sample_data(log_file):
    with open(log_file, "r") as file:
        for line in file:
            if line.startswith("n_in_sample:"):
                parts = line.split()
                n_in_sample = int(parts[1])
            elif line.startswith("n_out_of_sample:"):
                parts = line.split()
                n_out_of_sample = int(parts[1])
            elif line.startswith("n_invalid:"):
                parts = line.split()
                n_invalid = int(parts[1])

    return n_in_sample, n_out_of_sample, n_invalid


def plot_generalization_performance(sizes, data):
    """
    Generates a stacked line plot showing the in-sample, out-of-sample,
    and invalid cases for each dataset size.

    Parameters:
    - sizes: A list of dataset sizes (x-axis values).
    - data: A numpy array of shape (n_samples, 3), where each row contains
            [n_in_sample, n_out_of_sample, n_invalid].
    """

    # Split the data into the respective categories
    n_in_sample = data[:, 0]
    n_out_of_sample = data[:, 1]
    n_invalid = data[:, 2]

    # Create the stacked line plot
    fig, ax = plt.subplots()
    # ax.set_yscale('log')

    ax.stackplot(
        sizes,
        n_in_sample,
        n_out_of_sample,
        n_invalid,
        labels=["In Sample", "Out of Sample", "Invalid"],
    )

    # Add labels and title
    ax.set_xlabel("Training Dataset Size")
    ax.set_ylabel("Count")
    ax.set_title("Generalization Performance vs Dataset Size")
    ax.legend(loc="lower left")

    return fig


def main():
    log_files = [
        "log_16_dogs_worms.txt",
        "log_32_dogs_worms.txt",
        "log_64_dogs_worms.txt",
        "log_128_dogs_worms.txt",
        "log_256_dogs_worms.txt",
        "log_512_dogs_worms.txt",
        "log_1024_dogs_worms.txt",
        "log_2048_dogs_worms.txt",
        "log_4096_dogs_worms.txt",
    ]
    sizes = [
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
    ]

    data = np.array(
        [load_sample_data(log_file) for log_file in log_files],
        dtype=np.int32,
    )

    fig = plot_generalization_performance(sizes, data)
    fig.savefig("generalization.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
