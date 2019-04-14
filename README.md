# Programming and Scripting Project Set 2019

This repository is the submission for the 2019 project for the Programming and Scripting Module @ GMIT.

[Project is located:] (https://github.com/ianmcloughlin/project-pands/raw/master/project.pdf)
      
## To download this repository:

1. Go to GitHub
2. Go to my repository [https://github.com/mhurley100/pands-project.git]
3. Click the Clone or download button

## Research Summary
Ronald A. Fischer (British statistician and biologist) formulated linear discriminant analysis using a multivariate data set (Iris) in his 1936 paper "The use of multiple measurements in taxonomic problems". The Iris flowers were taken from the same location, measured by the same person and with the same instrument.

The data set comprises 50 samples from each of three species of Iris:
1. Iris setosa
2. Iris virginica
3. Iris versicolor

Four features are were measured from each sample:
1. Sepal length (cm)
2. Speal width (cm)
3. Petal length (cm)
4. Petal width (cm) 

The goal of discriminant analysis is given four measurements a flower can be classified correctly. 

Investigate:

1. Is there a correlation between sepal width and length?

2. Can an Iris species be identified by its dimension?

## Python Program (Iris-1.py)
Iris-1.py is a series of programs to analyse and interpret the data set.

    * The Iris flowers into their respective classes (setosa, versicolor and virginica).  
    * Pandas dataframe describe() is then used to analyse each data set
    * Statistical calculations are completed for each class (count, mean, std, min, 25%, 50%,      75% and max).
    * Graphics to aid understanding of the data by visualising and summarising.
        1.  The box and whisper plots demonstrate that distributions are relatively even
        2.  Histograms point out the diffences in the distributions
        3.  Scattergraph looks at the relationship between the attributes

3. Data Set Analysis

3. Summary of investigation
4. Supporting tables and graphics

##  References

https://www.kaggle.com/uciml/iris
http://archive.ics.uci.edu/ml/index.php
https://www.python.org/,
https://stackoverflow.com/,
https://matplotlib.org/
http://www.numpy.org/
https://pandas.pydata.org/
http://rcs.chemometrics.ru/Tutorials/classification/Fisher.pdf