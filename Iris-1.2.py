# Maura Hurley 28th April 2019
# This project documents and researches the Iris Data Set
import numpy as np

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
# Name each dataset - Virginica is the class Iris Virginica
petal_width = dataset[dataset['petal width']
np.arange(dataset)
array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
if petal_width == (0.0976-0.791),
  class = 'Iris-setosa'
if petal_width == (0.791,1.63] then class = 'Iris-versicolor'
if petal_width == (1.63,2.5] then class = 'Iris-virginica'