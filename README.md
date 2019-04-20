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
He picked 2 species of the Iris flowers from the same location, measured by the himself with the same instrument.
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
In the command line interface run Iris-1.py.

Iris-1.py is a python program that calculates statistics on the dataset.  The Iris raw data is imported into python as a csv file.  The Iris flowers are grouped into their respective classes (setosa, versicolor and virginica) and separated by petal width, petal length, sepal width and sepal length.  Each class of Iris within the data set is given a name.  
    - Iris-setosa
    - Iris-virginica
    - Iris-versicolor

Pandas dataframe describe() is used to analyse each data set.
- Describe() calculates statistics on the dataset.  Each class of Iris is grouped and statistics are completed (count, mean, std, min, 25%, 50%, 75% and max) for each class of Iris.

From the statisics generated in Iris-1.py it is evident that the standard deviation for all three species of Iris is low therfore the results indiciate that the data samples are reasonably close to the mean and potentially predictable.  The dataset is relatively small (150 samples) however it is still difficult to analyse using tables alone therefore we will use graphics to see if we can predict the species of Iris graphically.

### Graphics
Iris-2.py uses graphics to aid analysis of the data by correlating pictorally using the seaborn visualization library. This creates default pairs plot for quick examination of the data.

Firstly use seaborn pairplot to graphically compare the distributions.  A pairs plot allows us to see both distribution of single variables and relationships between two variables. Pair plots will help identify trends for follow-up analysis.   Iris-2.py is run Iris-Setosa stands out as being relatively easy to identify.
Pairsplots quickly explores distributions and relationships in a dataset. . A pairs plot is provides us with a comprehensive first look at the data.
    - 2 scatter plots are generated to demonstrate the correlation between sepal width and length and petal width and length for each class of Iris.  From the scatter matrix you can see that Iris Setosa is almost completely identifiable based on sepal width and length and petal width and length.  This is clealy demonstrated in the scatter plot matrices.
 
    Iris Versicolor and Iris Virginica are more alike as there is an overlap in sepal width and length but are more identifiable by petal width and length. 
    Therfore in this project I narrowing the scop of the project by focusing on a species of Iris looking at petal width and length with a view to further reducing as the analysis dictates.




### Data obsevations
The standard deviation for all three species of Iris is low therfore the results indiciate that the data is consistent.

n the attributes

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
- https://towardsdatascience.com
- https://pandas.pydata.org/
- https://seaborn.pydata.org/
- http://rcs.chemometrics.ru/Tutorials/classification/Fisher.pdf