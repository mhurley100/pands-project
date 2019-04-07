# Maura Hurley 28th April 2019
# This project documents and researches the Iris Data Set

# Modified from [https://machinelearningmastery.com/machine-learning-in-python-step-by-step/]

# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Load the dataset
# Modified using python tutorial 7.2.  [https://docs.python.org/3/tutorial/inputoutput.html]
# Open "Iris.csv" saved within this folder and call it f:
with open("Iris.csv", "r") as f:

# Import file as csv and the headers or "names" are assigned.  Modified from [https://stackoverflow.com/a/47111317]
  dataset = pd.read_csv(f)

# Modified from [https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation]
# Print the dataset so only first 20 appear
  print(dataset.head(20))

# Print number of rows for each Species.
print(dataset.groupby('class').size())

#Create 3 DataFrames for each class of Iris

# Modified from [https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation]
# Name each dataset - Setosa is the class Iris Setosa
setosa=dataset[dataset['class']=='Iris-setosa']
# Name each dataset - Versicolor is the class Iris Versicolor
versicolor =dataset[dataset['class']=='Iris-versicolor']
# Name each dataset - Virginica is the class Iris Virginica
virginica =dataset[dataset['class']=='Iris-virginica']
# Include headers for each dataset and separate by adding new line. Adapted from 
# [https://stackoverflow.com/a/45377991]. Call it newline
newline = ('\n')
# Print the class first ("Iris Setosa"), then use the pandas dataframe describe() to analyse the series
print('Iris Setosa',newline, setosa.describe())
# Print the class first ("Iris Versicolor"), then use the pandas dataframe describe() to analyse the series
print('Iris Versicolor',newline, versicolor.describe())
# Print the class first ("Iris Virginica"), then use the pandas dataframe describe() to analyse the series
print('Iris Virginica',newline, virginica.describe())