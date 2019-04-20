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
Edgar Anderson collected data on 3 different Iris species on the Gaspe Peninsula, Quebec, Canada.
Two of the three species were collected "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".[1] 
The data set comprises 50 samples from each of three species of Iris:
1. Iris setosa
2. Iris virginica
3. Iris versicolor

Four features are were measured from each sample:
1. Sepal length (cm)
2. Speal width (cm)
3. Petal length (cm)
4. Petal width (cm) 

Ronald A. Fischer (British statistician and biologist) used Anderson’s data and formulated linear discriminant analysis using the Iris dataset in his 1936 paper "The use of multiple measurements in taxonomic problems". 

The goal of discriminant analysis is given four measurements a flower can be classified correctly. This means that given certain data outcome may be predictable.


## Dataset Analysis
Is it possible to classify species of Iris (Setosa, Versicolor and Virginica) with given dimensions of sepal width, sepal length, petal width and petal length? Can classes be predicted given certain parameters?  This project will examine that possibility.

### Import data:

Iris-1.py is a python program that calculates statistics on the dataset.  The Iris raw data is imported into python as a csv file.  The Iris flowers are grouped into their respective classes (Setosa, Versicolor and Virginica) and separated by petal width, petal length, sepal width and sepal length.  Each class of Iris within the data set is given a name or class:  
- Iris-setosa
- Iris-virginica
- Iris-versicolor

Pandas dataframe describe() is used to analyse each data set.
- Describe() calculates statistics on the dataset.  Each class of Iris is grouped and statistics are completed (count, mean, std, min, 25%, 50%, 75% and max) for each class of Iris.

In the command line interface run Iris-1.py and observe the output.
View the output below:

From the statisics generated in Iris-1.py it is evident that the standard deviation for all three species of Iris is low therefore the results indiciate that the data samples are reasonably close to the mean and potentially predictable.  The dataset is relatively small (150 samples) however it is still complex and the next step is to graphically view the outputs.

### Graphics

Iris-2.py uses graphics to aid analysis of the dataset using the seaborn visualisation library.  This should help identify trends for follow-up analysis.
Seaborn pairplot is used to graphically compare the distributions.  A pairs plot provides  visualisation of both the distribution of single variables and relationships between two variables.    

In the command line interface run Iris-2.py.

When Iris-2.py is run Iris-Setosa stands out again as being relatively easy to classify given the 4 dimensions.
- 2 scatter plots are generated to demonstrate the correlation between sepal width and length and petal width and length for each class of Iris.  From the scatter matrix you can see that Iris Setosa is almost completely identifiable based on sepal width and length and petal width and length.  This is clealy demonstrated in the scatter plot matrices.
 
- Iris Versicolor and Iris Virginica are more alike as there is an overlap in sepal width and length but are more identifiable by petal width and length. 
  
In the command line interface run Iris-3.py.

This graphs a scatterplot for each pairwise relationship.  Again Setosa is very clearly distinguishable.  Versicolor and Virginica less so but should still be identifiable to a given acceptable range. 

Petal width:
- Iris-Virginica has the widest petals.  1.4 - 2.5 (1.8 to 2.5 is always Virginica)
- Iris-Versicolor has medium width petals. 1.0 - 1.8 is always Versicolor
- Iris-Setosa has the narrowest petals - 0.1 - 0.6 cm meaning Setosa can be identified using Petal width



## Applying Machine learning
Now that the dataset is understood, we can start implementing algorithms in machine learning. 

### Conclusion


Summary of investigation
## Supporting tables and graphics
 Species statistics:      |
##  References
- [1] Edgar Anderson (1935). "The irises of the Gaspé Peninsula". Bulletin of the American Iris Society. 59: 2–5.
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