# Maura Hurley 28th April 2019
# This project documents and researches the Iris Data Set

# Modified from [https://machinelearningmastery.com/machine-learning-in-python-step-by-step/]

# Load Pandas, pandas plotting, seaborn, matplotlib and csv libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# Load the dataset
# Modified using python tutorial 7.2.  [https://docs.python.org/3/tutorial/inputoutput.html]
# Open "Iris.csv" saved within this folder and call it f:
with open("Iris.csv", "r") as f:

# Import file as csv and the headers or "names" are assigned.  Modified from [https://stackoverflow.com/a/47111317]
  dataset = pd.read_csv(f)

# Adapted from [https://seaborn.pydata.org/tutorial/categorical.html]
sns.catplot(data=dataset, orient="h", kind="box");
# Adapted from [https://towardsdatascience.com/introduction-to-data-visualization-in-python-89a54c97fbed]
dataset.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=20)
# Display the charts
plt.show()