# Programming and Scripting Project 2019

## Table of Contents:

1.   Introduction
2.   Dataset Analysis
3.   Dataset Review
4.   Conclusion
5.   References

## Introduction
Edgar Anderson collected data on 3 different Iris species on the Gaspe Peninsula, Quebec, Canada.
Two of the three species were collected "all from the same pasture and picked on the same day and measured at the same time by the same person with the same apparatus".[1] 
The data set comprises 50 samples from each of three species of Iris:
1. Iris Setosa
2. Iris Virginica
3. Iris Versicolor

Four features are were measured from each sample:
1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm) 

Ronald A. Fischer (British statistician and biologist) used Anderson’s data and formulated linear discriminant analysis using the Iris dataset in his 1936 paper "The use of multiple measurements in taxonomic problems". 

The goal of discriminant analysis is given four measurements a flower can be classified correctly. Data analytics has evolved significantly since Fischer's paper with the Iris dataset used for training and testing of algorithms and Machine Learning.  It is possible that with multivariate samples outcomes can be predicted with almost 100% accuracy.

## Dataset Analysis
Is it possible to classify and predict species of Iris (Setosa, Versicolor and Virginica) with 4 given dimensions (sepal width, sepal length, petal width and petal length)? Classes can be predicted given multiple parameters.  This project examines univariate, multivariate and machine learning capabilities.   

### Import data:

Iris-1.py is a python program that calculates statistics on the dataset.  The Iris raw data is imported into python as a csv file.  The Iris flowers are grouped into their respective classes (Setosa, Versicolor and Virginica) and separated by petal width, petal length, sepal width and sepal length.  Each class of Iris within the data set is given a name or class:  
- Iris-Setosa
- Iris-Virginica
- Iris-Versicolor

Pandas dataframe describe() is used to analyse each data set.
- Describe() calculates statistics on the dataset.  Each class of Iris is grouped and statistics are completed (count, mean, std, min, 25%, 50%, 75% and max) for each class of Iris.

### Observe data:

In the command line interface run Iris-1.py and observe the output.
<<<<<<< HEAD
From the statistics generated in Iris-1.py it is evident that the standard deviation for all three species of Iris is low therefore the results indicate that the data samples are reasonably close to the mean and therefore predictable.  
=======
From the statisics generated in Iris-1.py it is evident that the standard deviation for all three species of Iris is low therefore the results indiciate that the data samples are reasonably close to the mean and therefore predictable.  
![Statistics](https://user-images.githubusercontent.com/47399526/56513217-a81c4400-6529-11e9-8d57-04c483dc3372.PNG)

Petal width stands out as an identifier for Iris class.  Petal width observations:
- Iris-Virginica has the widest petals.  1.4 - 2.5 (1.8 to 2.5 is always Virginica)
- Iris-Versicolor has medium width petals. 1.0 - 1.8 is always Versicolor
- Iris-Setosa has the narrowest petals - 0.1 - 0.6 cm meaning Setosa can be identified using petal width as the key parameter.

The dataset is relatively small (150 samples) however it is complex and the next step is to graphically view the output to see if any further insights are gained.


### Graphics
### Univariate Plots
Run Iris 1.1.py for univariate analysis where we explore one variable using box plots.  The box plots have vertical lines extending from the boxes (whiskers). These vertical lines indicate variability outside the upper and lower quartiles.
Setosa stands out from Virginica and Versicolor.  Setosa is easily identifiable by petal width (in particular) and petal length.  Virginica and Versicolor appear more closely related than Setosa.  Univariate plots display (similar to the tables above) that petal width is the key identifier.

![PL_Distribution_bp](https://user-images.githubusercontent.com/47399526/56511040-7e601e80-6523-11e9-8b19-5e16aff1d04a.PNG)
![PW_Distribution_bp](https://user-images.githubusercontent.com/47399526/56511041-7ef8b500-6523-11e9-9342-3fc2a5792f96.PNG)
![SL_Distribution_bp](https://user-images.githubusercontent.com/47399526/56511042-7ef8b500-6523-11e9-94f8-1e9a635c3191.PNG)
![SW_Distribution_bp](https://user-images.githubusercontent.com/47399526/56511043-7ef8b500-6523-11e9-92aa-f1092187f5e8.PNG)
### Multivariate Plots
In the command line interface run Iris-2.py.  Iris-2.py uses graphics to aid analysis and identification of trends within the dataset using the seaborn visualisation library.  Seaborn pairplot is used to graphically compare the distribution of each dimension and their relationship to other dimensions.    

Iris-Setosa stands out as being easy to classify given the 4 dimensions.
- 2 scatter plots are generated to demonstrate the correlation between sepal width and length and petal width and length for each class of Iris.  From the scatter matrix you can see that Iris Setosa is almost completely identifiable based on sepal width and length and petal width and length.  This is clearly demonstrated in the scatter plot matrices.
 
- Iris Versicolor and Iris Virginica are more alike as there is an overlap in sepal width and length but are more identifiable by petal width and length. 
  ![Seaborn1_Distribution](https://user-images.githubusercontent.com/47399526/56510680-6b008380-6522-11e9-8319-2c838b5ff75c.PNG)
In the command line interface run Iris-3.py.

This graphs a scatterplot for each pairwise relationship.  Again Setosa is very clearly distinguishable.  Versicolor and Virginica less so but should still be identifiable to a given acceptable range. 
![Seaborn2_Distribution](https://user-images.githubusercontent.com/47399526/56510843-ee21d980-6522-11e9-92e1-92158d57c163.PNG)

Multivariate plotting lines for all 4 dimensions enables visualisation of the relationship between the variables.  Again, the Setosa easily identifiable as being different.  Multivariate plotting is adding a new dimension and is a step above tables and univariate analysis.

### Applying Machine learning
Machine learning techniques aid validation of the dataset and estimate accuracy.  KNN (K-Nearest Neighbor) appears to be the classification algorithm of choice for assigning a class to new data point. The main advantage of KNN is that it does not make any assumptions on the data distribution. It keeps all the training data to make future predictions by computing the similarity between an input sample and each training instance.

 Each instance describes the properties of an observed flower measurements and the output variable is specific iris species.

Run Iris-4.py (program adapted from machinelearningmastery.com).  KNN can be used for predicting class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its nearest K neighbours. It can also be used for regression — output is the value for the object (predicts continuous values). This value is the average (or median) of the values of its k nearest neighbours.
Accuracy is 90% which gives high confidence.  This is a step change in data analytics, machine learning can predict outcomes.

### Dataset Review

Data analytics has evolved with the advent of technology and analytical tools that can predict outcomes with high levels of accuracy. However, the Iris dataset can be analysed almost as effectively with only one variable - petal width. Using multiple variables makes simple analysis more difficult to explain and critique.  Also, it has been established that a variety of classification methods yield good results on Iris however Iris is predictable and relatively easy to classify.  

Data analysis on a dataset is only as good as the raw data input.  Versicolor and Virginica are so alike - should additional samples have been taken to further improve and enhance the result instead of applying complex machine learning models to try to predict classification?  It may be that there are issues with the underlying data.  Two of the three samples were picked on the same day - this is a qualitative variable which is not factored into any of the analytical tools.

### Conclusion

There are relatively few features in the Iris dataset and two of the classes are much more correlated than the third.  Setosa is very easily separated by the petal width and petal length. Therefore, data modelling lends itself in this case to good results.  We can predict Setosa with almost 100% accuracy given known data points.


In Machine Learning, there is no specific model or an algorithm which can give 100% result to every single dataset. We need to understand the data before we apply any algorithm and build our model depending on the desired result. 

More basic modelling is sufficient as the Iris dataset is predictable.  However, machine learning has undoubted benefits given complexity.

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
- https://www.tutorialspoint.com/
