# Programming and Scripting Project 2019

## Table of Contents:

1.   Introduction
2.   Background Research
3.   Dataset Analysis
4.   Comparative Analysis
5.   Conclusion
6.   References

## Introduction:
This repository is the 2019 project submission for the Programming and Scripting Module @ GMIT.

Project is located [https://github.com/mhurley100/pands-project/blob/master/project%20(Iris).pdf]
Iris data set is saved within the repository as a text file [https://github.com/mhurley100/pands-project/blob/master/Project%20Iris.txt]
      
### To download this repository:

1. Go to GitHub
2. Go to my repository [https://github.com/mhurley100/pands-project.git]
3. Click the Clone or download button

## Background Research
Edgar Anderson collected data on 3 different iris species on the Gaspe Peninsula, Quebec, Canada.
He picked the Iris flowers from the same location, measured by the himself with the same instrument.
The data set comprised 50 samples from each of three species of Iris:
1. Iris setosa
2. Iris virginica
3. Iris versicolor

Four features are were measured from each sample:
1. Sepal length (cm)
2. Speal width (cm)
3. Petal length (cm)
4. Petal width (cm) 

Ronald A. Fischer (British statistician and biologist) used Anderson’s data and formulated linear discriminant analysis using the Iris dataset in his 1936 paper "The use of multiple measurements in taxonomic problems". 

The goal of discriminant analysis is given four measurements a flower can be classified correctly. Given certain data outcome can be predicted.

Expectations of this analysis

1. Is there a correlation between sepal width and length?

2. Can an Iris species be identified by its dimensions?

## Dataset Analysis

### Import data:
Iris-1.py is a series of python commands that analyses and interprets the dataset.  The Iris raw data is imported into python as a csv file.  The Iris flowers are grouped into their respective classes (setosa, versicolor and virginica) and separated by petal width, petal length, sepal width and sepal length.  Each class of Iris within the data set is given a name.  
    - Iris-setosa
    - Iris-virginica
    - Iris-versicolor

### Analyse Data using pandas
- Pandas dataframe describe() is then used to analyse each data set
- Statistical calculations are completed for each class (count, mean, std, min, 25%, 50%, 75% and max) for each class of Iris.
A low standard deviation indicates that the data points tend to be close to the mean (also called the expected value) of the set, while a high standard deviation indicates that the data points are spread out over a wider range of values.

- Graphics to aid understanding of the data by visualising and summarising.
    - The box and whisper plots demonstrate that distributions are relatively even.
    - Histograms point out the diffences in the distributions
    - Scattergraph looks at the relationship between the attributes

- Data Set Analysis

Summary of investigation
## Supporting tables and graphics
 Species statistics:      |
##  References

- https://www.kaggle.com/uciml/iris
- http://archive.ics.uci.edu/ml/index.php
- https://www.python.org/,
- https://stackoverflow.com/,
- https://matplotlib.org/
- http://www.numpy.org/
- https://pandas.pydata.org/
- http://rcs.chemometrics.ru/Tutorials/classification/Fisher.pdf