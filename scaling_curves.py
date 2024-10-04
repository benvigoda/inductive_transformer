import os
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore

# We have loss vs. #steps to plot for different numbers of training sentences
# Each data set is stored in a different file:
num_training_sentences = [8]
file_names = [f"{n}_loss.csv" for n in num_training_sentences]
# Each file has two columns: steps and loss
# The steps is an integer, and the loss is a float
data_sets = []
for file_name in file_names:
    data_set = pd.read_csv(file_name)
    # Limit to the first 400 steps
    data_set = data_set#[data_set["Steps"] <= 400]
    data_sets.append(data_set)

# Plot each data set
# on the same figure, in different colors
# connecting the dots with lines of the same color as the dots
# and making the dots a bit transparent
# with a legend showing the number of training sentences
plt.figure(figsize=(10, 6))
for i, data_set in enumerate(data_sets):
    plt.plot(data_set["Steps"], data_set["Loss"], marker="o", label=f"{num_training_sentences[i]} sentences", alpha=0.5)

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs. Steps for Different Numbers of Training Sentences")
plt.show()
