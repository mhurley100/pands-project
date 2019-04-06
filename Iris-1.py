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
# Name the headers of the dataset
  headers = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# Import file as csv and the headers or "names" are assigned.  Modified from [https://stackoverflow.com/a/47111317]
  dataset = pd.read_csv(f, names=headers)
  
# Print the dataset
  print(dataset)
  