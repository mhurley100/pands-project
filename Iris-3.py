# Maura Hurley 28th April 2019
# This project documents and researches the Iris Data Set

# Load Pandas, matplotlib.pyplot, seaborn and csv libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
import csv

# Load the dataset
# Modified using python tutorial 7.2.  [https://docs.python.org/3/tutorial/inputoutput.html]
# Open "Iris.csv" saved within this folder and call it f:
with open("Iris.csv", "r") as f:

# Import file as csv and the headers or "names" are assigned.  Modified from [https://stackoverflow.com/a/47111317]
  dataset = pd.read_csv(f)

# Modified from [https://seaborn.pydata.org/generated/seaborn.PairGrid.html]
# Use seaborn pairgrid functionality on the dataset
g = sns.PairGrid(dataset, hue="class")
# Generate histogram using seaborn
g = g.map_diag(plt.hist, histtype="step", linewidth=3)
# Generate scatter plot using seaborn
g = g.map_offdiag(plt.scatter)
# Add legend
g.add_legend()
# Show the plots
plt.show()