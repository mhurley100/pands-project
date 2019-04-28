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

#Create 3 DataFrames for each class of Iris

# Modified from [https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation]
# Name each dataset - Setosa is the class Iris Setosa
setosa = dataset[dataset['class']=='Iris-setosa']
# Name each dataset - Versicolor is the class Iris Versicolor
versicolor = dataset[dataset['class']=='Iris-versicolor']
# Name each dataset - Virginica is the class Iris Virginica
virginica = dataset[dataset['class']=='Iris-virginica']

# Use seaborn to generate graphics [https://seaborn.pydata.org/generated/seaborn.pairplot.html]
g = sns.pairplot(dataset, hue='class', markers=["o", "s", "D"])
# Show the plot
plt.show()

# Modified from [https://seaborn.pydata.org/generated/seaborn.PairGrid.html]
# Use seaborn pairgrid functionality on the dataset
g = sns.PairGrid(dataset, hue="class", palette="Set2", hue_kws={"marker": ["o", "s", "D"]})
# Generate scatter plot using seaborn
g = g.map(plt.scatter, linewidths=1, edgecolor="w", s=40)
# Add legend
g = g.add_legend()
# Show the plots
plt.show()