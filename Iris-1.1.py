# Maura Hurley 28th April 2019
# This project documents and researches the Iris Data Set

# Modified from [https://machinelearningmastery.com/machine-learning-in-python-step-by-step/]

# Load Pandas, seaborn, matplotlib and csv libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# Load the dataset
# Modified using python tutorial 7.2.  [https://docs.python.org/3/tutorial/inputoutput.html]
# Open "Iris.csv" saved within this folder and call it f:
with open("Iris.csv", "r") as f:

# Import file as csv and the headers or "names" are assigned.  Modified from [https://stackoverflow.com/a/47111317]
  dataset = pd.read_csv(f)

# Adapted from [https://www.tutorialspoint.com/seaborn/seaborn_quick_guide.htm] and
# https://stackoverflow.com/a/42409861

sns.boxplot(x="class", y="sepal length", data=dataset).set_title('Compare Sepal Length Distribution')
# Show the plot
plt.show()
sns.boxplot(x="class", y="sepal width", data=dataset).set_title('Compare Sepal Width Distribution')
# Show the plot
plt.show()
sns.boxplot(x="class", y="petal length", data=dataset).set_title('Compare Petal Length Distribution')
# Show the plot
plt.show()
sns.boxplot(x="class", y="petal width", data=dataset).set_title('Compare Petal Width Distribution')
# Show the plot
plt.show()

sns.violinplot(x="class", y="sepal length", data=dataset, size=6).set_title('Compare Sepal Length Distribution')
# Show the plot
plt.show()
sns.violinplot(x="class", y="sepal width", data=dataset, size=6).set_title('Compare Sepal Width Distribution')
# Show the plot
plt.show()
sns.violinplot(x="class", y="petal length", data=dataset, size=6).set_title('Compare Petal Length Distribution')
# Show the plot
plt.show()
sns.violinplot(x="class", y="petal width", data=dataset, size=6).set_title('Compare Petal Width Distribution')
# Show the plot
plt.show()