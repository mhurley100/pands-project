# Maura Hurley 28th April 2019
# This project documents and researches the Iris Data Set

# Modified from [https://machinelearningmastery.com/machine-learning-in-python-step-by-step/]

# Load Pandas and csv libraries
import pandas as pd
import csv
from tabulate import tabulate

# Load the dataset
# Modified using python tutorial 7.2.  [https://docs.python.org/3/tutorial/inputoutput.html]
# Open "Iris.csv" saved within this folder and call it f:
with open("Iris.csv", "r") as f:

# Import file as csv and the headers or "names" are assigned.  Modified from [https://stackoverflow.com/a/47111317]
  dataset = pd.read_csv(f)

# Modified from [https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation]
# Print the dataset so only first 20 appear
  print(dataset.head(20))

# Print summary dataset for each class
print(dataset.groupby('class').size())

#Create 3 DataFrames for each class of Iris
# Modified from [https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation]
# Name each dataset - Setosa is the class Iris Setosa
setosa = dataset[dataset['class']=='Iris-setosa']
# Name each dataset - Versicolor is the class Iris Versicolor
versicolor = dataset[dataset['class']=='Iris-versicolor']
# Name each dataset - Virginica is the class Iris Virginica
virginica = dataset[dataset['class']=='Iris-virginica']

# Format for github.  Adapted from [https://bitbucket.org/astanin/python-tabulate] 
# Name the headers for the printed output to paste into github
headers={"sepal length", "sepal width","petal length","petal width"}
# Print the class first ("Iris Setosa"), then use the pandas dataframe describe() to analyse the series
print(tabulate(setosa.describe(), headers,tablefmt="github"))
# Print the class first ("Iris Versicolor"), then use the pandas dataframe describe() to analyse the series
print(tabulate(versicolor.describe(), headers,tablefmt="github"))
# Print the class first ("Iris Virginica"), then use the pandas dataframe describe() to analyse the series
print(tabulate(virginica.describe(), headers,tablefmt="github"))