# Maura Hurley 28th April 2019
# This project documents and researches the Iris Data Set

# Adapted from [https://machinelearningmastery.com/machine-learning-in-python-step-by-step/]

# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import csv

# Load the dataset
# Modified using python tutorial 7.2.  [https://docs.python.org/3/tutorial/inputoutput.html]
# Open "Iris.csv" saved within this folder and call it f:
with open("Iris_lf.csv", "r") as f:

# Import file as csv and the headers or "names" are assigned.  Modified from [https://stackoverflow.com/a/47111317]
  dataset = pd.read_csv(f)

# Adapted from [https://machinelearningmastery.com/machine-learning-in-python-step-by-step/]
# Split-out validation dataset
array = dataset.values
X = array[:,0:5]
Y = array[:,5]
# Use 20% of dataset for testing
validation_size = 0.20
# random seed
seed = 7
# Training data is in X_train and Y_train for model training and X_validation and Y_validation for later use
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Make predictions on validation dataset.  Adapted from [https://machinelearningmastery.com/machine-learning-in-python-step-by-step/]
# KNeighborsClassifier() is referred to as knn
knn = KNeighborsClassifier()
# Training samples
knn.fit(X_train, Y_train)
# knn.predict(X_validation) is referred to as predictions
predictions = knn.predict(X_validation)
# Print accuracy score
print(accuracy_score(Y_validation, predictions))
# Print confusion matrix
print(confusion_matrix(Y_validation, predictions))
# Print classification report
print(classification_report(Y_validation, predictions))