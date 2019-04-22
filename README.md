# Programming and Scripting Project 2019

## Table of Contents:

1.   Introduction
2.   Dataset Analysis
3.   Comparative Analysis
4.   Conclusion
5.   References

## Introduction
Edgar Anderson collected data on 3 different Iris species on the Gaspe Peninsula, Quebec, Canada.
Two of the three species were collected "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".[1] 
The data set comprises 50 samples from each of three species of Iris:
1. Iris Setosa
2. Iris Virginica
3. Iris Versicolor

Four features are were measured from each sample:
1. Sepal length (cm)
2. Speal width (cm)
3. Petal length (cm)
4. Petal width (cm) 

Ronald A. Fischer (British statistician and biologist) used Anderson’s data and formulated linear discriminant analysis using the Iris dataset in his 1936 paper "The use of multiple measurements in taxonomic problems". 

The goal of discriminant analysis is given four measurements a flower can be classified correctly. Data analytics has evolved significantly since Fischer's paper with the Iris dataset used for training and testing of algorithms and Machine Learning.  It is possible that with multivariate samples outcomes can be predicted with almost 100% accuracy.

## Dataset Analysis
Is it possible to classify and predict species of Iris (Setosa, Versicolor and Virginica) with 4 given dimensions (sepal width, sepal length, petal width and petal length)? Classes be predicted given multiple parameters.  This project examines univariate, multivariate and machine learning capabilities.   

### Import data:

Iris-1.py is a python program that calculates statistics on the dataset.  The Iris raw data is imported into python as a csv file.  The Iris flowers are grouped into their respective classes (Setosa, Versicolor and Virginica) and separated by petal width, petal length, sepal width and sepal length.  Each class of Iris within the data set is given a name or class:  
- Iris-setosa
- Iris-virginica
- Iris-versicolor

Pandas dataframe describe() is used to analyse each data set.
- Describe() calculates statistics on the dataset.  Each class of Iris is grouped and statistics are completed (count, mean, std, min, 25%, 50%, 75% and max) for each class of Iris.

### Observe data:

In the command line interface run Iris-1.py and observe the output.
From the statisics generated in Iris-1.py it is evident that the standard deviation for all three species of Iris is low therefore the results indiciate that the data samples are reasonably close to the mean and therefore predictable.  

Petal width stands out as an identifier for Iris class.  Petal width observations:
- Iris-Virginica has the widest petals.  1.4 - 2.5 (1.8 to 2.5 is always Virginica)
- Iris-Versicolor has medium width petals. 1.0 - 1.8 is always Versicolor
- Iris-Setosa has the narrowest petals - 0.1 - 0.6 cm meaning Setosa can be identified using petal width as the key parameter.

The dataset is relatively small (150 samples) however it is complex and the next step is to graphically view the output to see if any further insights are gained.


### Graphics
### Univariate Plots
Run Iris 1.1.py for univariate analysis where we explore one variable.
Box plots and histograms.  The box plots have vertical lines extending from the boxes (whisker). These vertical lines indicate variability outside the upper and lower quartiles.
Setosa stands out from Virginica and Versicolor.  Setosa is easily identifiable by petal width (in particular) and petal length.  Virginica and Versicolor appear more closely related than Setosa.  Univariate plots display (similar to the tables above) that petal width is the key identifier.

### Multivariat Plots
In the command line interface run Iris-2.py.  Iris-2.py uses graphics to aid analysis and identification of trends within the dataset using the seaborn visualisation library.  Seaborn pairplot is used to graphically compare the distribution of each dimension and their relationship to other dimensions.    

Iris-Setosa stands out as being easy to classify given the 4 dimensions.
- 2 scatter plots are generated to demonstrate the correlation between sepal width and length and petal width and length for each class of Iris.  From the scatter matrix you can see that Iris Setosa is almost completely identifiable based on sepal width and length and petal width and length.  This is clealy demonstrated in the scatter plot matrices.
 
- Iris Versicolor and Iris Virginica are more alike as there is an overlap in sepal width and length but are more identifiable by petal width and length. 
  
In the command line interface run Iris-3.py.

This graphs a scatterplot for each pairwise relationship.  Again Setosa is very clearly distinguishable.  Versicolor and Virginica less so but should still be identifiable to a given acceptable range. 

Multivariate plotting lines up all 4 dimensions and makes comparing the inter relationship of the variables straightforward.  However no further insights are garnered as the Setosa is so easy to distinguish from viewing tables and univariate analysis.

## Applying Machine learning
Machine learning techniques are freely available to aid validation of the data and estimate accuracy.  The Iris data set is so widely used, it needs mentioning.  

KNN (K-Nearest Neighbor) appears to be the classification algorithm of choice for assigning a class to new data point. The main advantage of KNN is that it does not make any assumptions on the data distribution. It keeps all the training data to make future predictions by computing the similarity between an input sample and each training instance.

 Each instance describes the properties of an observed flower measurements and the output variable is specific iris species.

Run Iris-4.py (machinelearningmastery.com).  KNN can be used for classification — the output is a class membership (predicts a class — a discrete value). An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors. It can also be used for regression — output is the value for the object (predicts continuous values). This value is the average (or median) of the values of its k nearest neighbors.

### Conclusion
Data analytics has evolved with the advent of technology and analytical tools that can predict outcomes with machine learning. 

However, the dataset could have be analysed almost as effectively with only one variable - petal width. Using multiple variables makes simple analysis more dificult to explain and comprehend.

In Machine Learning, there is no specific model or an algorithm which can give 100% result to every single dataset. We need to understand the data before we apply any algorithm and build our model depending on the desired result. 

Is data analytics making data more difficult to explain i.e. does it make comprehension of the underlying data more difficult?  
It has been well established that a variety of classification models yield incredibly good results on Iris however Iris is predictable.  Data analysis is only as good as the raw data input.  Versicolor and Virginica are so alike - should additional samples have been taken to further improve and enhance the result instead of applying complex machine learning models to try to predict classification

Secondly, we can observe that there are relatively few features in the Iris dataset. Moreover, if you look at the dataset description you can see that two of the features are very highly correlated with the class outcomes.

These correlation values are linear, single-feature correlations, which indicates that one can most likely apply a linear model and observe good results. N

Taking these facts into account, that (a) there are few features to begin with and (b) that there are high linear correlations with class, would all point to a less complex, linear function as being the appropriate predictive model-- by using a single hidden node, you are very nearly using a linear model.

It can also be noted that, in the absence of any hidden layer (i.e., just input and output nodes), and when the logistic transfer function is used, this is equivalent to logistic regression

     |
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
- https://machinelearningmastery.com