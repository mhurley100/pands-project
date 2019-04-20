# Maura Hurley 28th April 2019
# This project documents and researches the Iris Data Set

# Modified from [https://machinelearningmastery.com/machine-learning-in-python-step-by-step/]

# Load Pandas, scatter matrix from pandas, matplotlib and csv libraries
import pandas as pd
from pandas.plotting import scatter_matrix
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

# Use box plot to display. Adapted from [https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/]

setosa.boxplot()
plt.show()
versicolor.boxplot()
plt.show()
virginica.boxplot()
plt.show()
# Create histograms using pandas dataframe hist ()
# Adapted from [https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/]

setosa.groupby('class').hist()
plt.show()
versicolor.groupby('class').hist()
plt.show()
virginica.groupby('class').hist()
plt.show()

from pandas.plotting import scatter_matrix
scatter_matrix(setosa, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.title('Iris Dataset scatter Plot')
plt.show()
