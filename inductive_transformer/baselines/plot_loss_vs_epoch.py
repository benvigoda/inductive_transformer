import click
import numpy as np
import matplotlib.pyplot as plt


def load_loss_data(log_file):
    """Loads the loss data from the log file and returns a numpy array of (epoch, train_loss, test_loss)."""

    epochs = []
    losses = []
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

                # Parse epoch, train loss, and test loss
                epoch, train_loss, test_loss = map(float, line.split())
                epochs.append(epoch)
                losses.append([train_loss, test_loss])

    # We expect the epochs to be in order
    epochs = np.array(epochs, dtype=np.int32)
    n_epochs = len(epochs)
    assert np.all(epochs == np.arange(n_epochs))

    return np.array(losses, dtype=np.float64)


def plot_losses(all_losses, keys):
    """Plots training and test losses for multiple datasets on the same figure."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, losses in enumerate(all_losses):
        epochs = np.arange(losses.shape[0])
        train_losses = losses[:, 0]
        test_losses = losses[:, 1]

        # Plot each dataset's losses with a unique label
        key = keys[i]
        ax.plot(epochs, train_losses, label=f"{key} sentences training loss")
        ax.plot(epochs, test_losses, label=f"{key} sentences full loss")

    # Set labels, title, and grid
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Test Loss vs. Epoch (Multiple Runs)")
    ax.legend()
    ax.grid(True)

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
    keys = [
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

    all_losses = [load_loss_data(log_file) for log_file in log_files]
    all_losses = [losses[:1000, :] for losses in all_losses]

    fig = plot_losses(all_losses, keys)
    fig.savefig("loss_vs_epoch.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
