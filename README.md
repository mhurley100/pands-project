# Programming and Scripting Project 2019
![Iris flowers](https://user-images.githubusercontent.com/47399526/56853734-a5539180-6923-11e9-8fab-da93703ca17c.PNG)
# Iris Dataset Analysis

This project investigates and analyses the Iris data set using python.

Project plan is as follows:
1. Research the Iris Dataset listing references to evidence that research.
2. Use python to analyse the Iris data set.
3. Summarise finding using supporting tables and graphics as relevant.
4. Find an interesting angle to pursue and investigate.

## Table of Contents:
1.   Introduction
2.   Dataset Analysis (including statistics, univariate & mulitvariate plots)
3.   Insights and Comparative Analysis
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

Ronald A. Fischer (British statistician and biologist) used Anderson’s data and formulated linear discriminant analysis using the Iris dataset in his 1936 paper "The use of multiple measurements in taxonomic problems". The goal of discriminant analysis is given four measurements a flower can be classified correctly. Data analytics has evolved significantly since Fischer's paper with the Iris dataset used for training and testing of algorithms and Machine Learning.  It appears possible that with multivariate samples outcomes can be predicted with very high levels of accuracy.

## Dataset Analysis
Is it possible to classify and predict species of Iris (Setosa, Versicolor and Virginica) with 4 given dimensions (sepal width, sepal length, petal width and petal length)? Can classes can be predicted given multiple parameters?    

#### Statistics
Iris-1.py is a python program that calculates basic statistics on the dataset.  The Iris raw data is imported into python as a csv file.  

Observations of the dataset:
- It is small - only 150 rows and 4 features.
- It appears straightforward with no data missing.

There are so many online sources, however many sites have the same repeated analysis.  Machinelearningmastery.com [2] and kaggle.com [4] are used as learning resources and training tools for Iris-1.py. The Iris data set can be downloaded from multiple libraries.  I chose to save as a csv file within this repository. I also had to import tabulate by running "tabulate" on the command line interface.  I researched how to formulate a table in python and import into github using (itbucket.org [14] & pypi.org/project/tabulate [13])  The result of this research is as follows:

|Setosa    |   sepal length |   petal width |   sepal width |   petal length |
|----------|----------------|---------------|---------------|----------------|
| count    |       50       |     50        |     50        |       50       |
| mean     |        5.006   |      3.418    |      1.464    |        0.244   |
| std      |        0.35249 |      0.381024 |      0.173511 |        0.10721 |
| min      |        4.3     |      2.3      |      1        |        0.1     |
| 25%      |        4.8     |      3.125    |      1.4      |        0.2     |
| 50%      |        5       |      3.4      |      1.5      |        0.2     |
| 75%      |        5.2     |      3.675    |      1.575    |        0.3     |
| max      |        5.8     |      4.4      |      1.9      |        0.6     |

|Versicolor|   sepal length |   petal width |   sepal width |   petal length |
|----------|----------------|---------------|---------------|----------------|
| count    |      50        |     50        |     50        |      50        |
| mean     |       5.936    |      2.77     |      4.26     |       1.326    |
| std      |       0.516171 |      0.313798 |      0.469911 |       0.197753 |
| min      |       4.9      |      2        |      3        |       1        |
| 25%      |       5.6      |      2.525    |      4        |       1.2      |
| 50%      |       5.9      |      2.8      |      4.35     |       1.3      |
| 75%      |       6.3      |      3        |      4.6      |       1.5      |
| max      |       7        |      3.4      |      5.1      |       1.8      |

|Virginica |   sepal length |   petal width |   sepal width |   petal length |
|----------|----------------|---------------|---------------|----------------|
| count    |       50       |     50        |     50        |       50       |
| mean     |        6.588   |      2.974    |      5.552    |        2.026   |
| std      |        0.63588 |      0.322497 |      0.551895 |        0.27465 |
| min      |        4.9     |      2.2      |      4.5      |        1.4     |
| 25%      |        6.225   |      2.8      |      5.1      |        1.8     |
| 50%      |        6.5     |      3        |      5.55     |        2       |
| 75%      |        6.9     |      3.175    |      5.875    |        2.3     |
| max      |        7.9     |      3.8      |      6.9      |        2.5     |


 Anaconda's Pandas tool is the tool of choice for statistical analysis.  It is very powerful as it manipulates, analyses and performs statistical analysis with minimal coding.  Iris-1.py contains pandas python programs to group the Iris flowers into their respective classes (Setosa, Versicolor and Virginica) and separate by petal width, petal length, sepal width and sepal length.  Pandas dataframe describe() was used to perform this analysis.  Below are the programs used to analyse the dataset:

1.  The first 20 lines are printed by running the following line of code on the dataset:
    -(print(dataset.head(20)))
2.  The dataset is printed, grouped by class (species) and sample size using the following code:
    -print(dataset.groupby('class').size())
3.  3 dataframes are created for each species ('class') of Iris and each named as their respective species using the following code:
    - e.g.setosa is separated into its class as follows: dataset[dataset['class']=='Iris-setosa']. 
4.  Pandas dataframe describe() calculates statistics on the dataset.  Each class of Iris is grouped and statistics are completed (count, mean, std, min, 25%, 50%, 75% and max) for each class of Iris using tabulate to enable to paste into github as follows:
    -print(tabulate(setosa.describe(), headers,tablefmt="github"))

### Data Analysis:
The standard deviation for all three species of Iris is low therefore the results indicate that the data samples are reasonably close to the mean and as a result potentially predictable.
#### Observations:
- Petal width stands out as an identifier for Iris class.  
- Petal width observations:
  - Iris-Virginica has the widest petals.  1.4 - 2.5 (1.8 to 2.5 is always Virginica)
  - Iris-Versicolor has medium width petals. 1.0 - 1.8 is always Versicolor
  - Iris-Setosa has the narrowest petals - 0.1 - 0.6 cm meaning Setosa can be identified using petal width as the key parameter.


### Univariate Plots
Univariate plots help understand each variable independently by analysing interactions between variables.  There is a wide array of plotting resources available online from historgrams, barcharts, scatter graphs, box, violin, linear etc all producing similar results.  I chose box plots and violin plots.  Box plots summarise within and between groups using 25th, 50th & 75th percentiles meaning that they are not influenced by outliers.  Violin plots show the probability distribution of the sample by computing empirical distributions using kernal density estimation (KDE) (source matplotlib.org[8]).  I used Seaborn as it has the added advantage of colour and sizing options.    

Machinelearningmastery.com [2], tutorialspoint.com [7], seaborn.pydata.org [9], stackoverflow.com[5] and python.org [6]are used as learning resources and training tools for Iris-1.1.py.  

Pandas, seaborn, matplotlib and csv libraries are imported.  Run Iris 1.1.py where the four variables of petal length, petal width, sepal length and petal width are explored using box and violin plots.  

Program outputs:
1.  Box plot for petal length, petal width, sepal length and petal width sepal length are displayed using the following code (e.g Sepal Length) with the relevant header included:
    - sns.boxplot(x="class", y="sepal length", data=dataset).set_title('Compare Sepal Length Distribution')
2. Violin plot for each variable (petal length, petal width, sepal length and petal width sepal length).  E.g sepal length below with the relevant header included:
    - sns.violinplot(x="class", y="sepal length", data=dataset, size=6).set_title('Compare Sepal Length Distribution')
3.  Show the plot.  The box plot is displayed using the following code:
    - plt.show()

### Data Analysis:
The box plots have vertical lines extending from the boxes. These vertical lines indicate variability outside the upper and lower quartiles.  Setosa stands out from Virginica and Versicolor.  Setosa is easily identifiable by petal width (in particular) and petal length.  Virginica and Versicolor appear more closely related than Setosa.  A violin plot shows the density of the data.  Using violin plots Setosa stands out due to its density across all 4 variables.  This makes its easy to identify.  Univariate plots display (similar to the tables above) that petal width is the key identifier and Setosa is different from the other 2 species.  



### Multivariate Plots
Machinelearningmastery.com [2], www.kaggle.com [4], stackoverflow.com[5] seaborn.pydata.org [9], http://www.learn4master.com [12] and python.org [6] were used as learning resources and training tools for Iris-2.py. and Iris-3.py.  Pairplots were used to analyse the relationship between each variable and also explain the relationship between variables.  

Pandas, seaborn, matplotlib and csv libraries are imported with seaborn.pydata.org,kaggle.com[7] and stackoverflow.com[5] used as resources.

In the command line interface run Iris-2.py & Iris-3.py.  Seaborn pairplot is used to graphically compare the distribution of each dimension and their relationship to other dimensions.

Program outputs:
1.  3 dataframes are created for each species ('class') of Iris and each named as their respective species using the following code:
    - e.g.setosa - dataset[dataset['class']=='Iris-setosa']. 
2.  Pair plot for petal length, petal width, sepal length and petal width sepal length are displayed using the following code (e.g Sepal Length):
    - g = sns.pairplot(dataset, hue='class', markers=["o", "s", "D"])
3.  Use seaborn pairgrid functionality on the dataset:
g = sns.PairGrid(dataset, hue="class", palette="Set2", hue_kws={"marker": ["o", "s", "D"]})
Generate scatter plot using seaborn:
g = g.map(plt.scatter, linewidths=1, edgecolor="w", s=40)
4.  Show the plot.  The box plot is displayed using the following code:
    plt.show()

### Data Analysis:
Iris-Setosa stands out as being easy to classify given the 4 dimensions.
- 2 scatter plots are generated to demonstrate the correlation between sepal width and length and petal width and length for each class of Iris.  From the scatter matrix you can see that Iris Setosa is almost completely identifiable based on sepal width and length and petal width and length.  This is clearly demonstrated in the scatter plot matrices.
 
- Iris Versicolor and Iris Virginica are more alike as there is an overlap in sepal width and length but are more identifiable by petal width and length. 
  ![Seaborn1_Distribution](https://user-images.githubusercontent.com/47399526/56510680-6b008380-6522-11e9-8319-2c838b5ff75c.PNG)
In the command line interface run Iris-3.py.

This graphs a scatterplot for each pairwise relationship.  Again Setosa is very clearly distinguishable.  Versicolor and Virginica less so but should still be identifiable to a given acceptable range. 
![Seaborn2_Distribution](https://user-images.githubusercontent.com/47399526/56510843-ee21d980-6522-11e9-92e1-92158d57c163.PNG)

Multivariate plotting lines for all 4 dimensions enables visualisation of the relationship between the variables.  Again, the Setosa easily identifiable as being different.  Multivariate plotting is adding a new dimension and is a step above tables and univariate analysis.

### Applying Machine learning
Machine learning techniques aid validation of the dataset and estimate accuracy. It involves learning properties of a data set and then testing those properties against another data set. A common practice in machine learning is to evaluate an algorithm by splitting a data set into two- one of those sets the training set (properties are learned) and the other the testing set (to test the learned properties) [10] https://scikit-learn.org/stable/tutorial/basic/tutorial.html.

KNN (K-Nearest Neighbor) appears to be the classification algorithm of choice for assigning a class to new data point. The main advantage of KNN is that it does not make any assumptions on the data distribution. It keeps all the training data to make future predictions by computing the similarity between an input sample and each training instance.

 Each instance describes the properties of an observed flower measurements and the output variable is specific iris species.

Run Iris-4.py (program adapted from machinelearningmastery.com[2]).  

Pandas, pandas plotting, matplotlib, csv and several sklearn libraries are imported.  

Program outputs:
1.  Iris Dataset is defined as an array with the below code (converts list by calling the array() function)
array = dataset.values
2. X & Y arrays are sliced:
X = array[:,0:4]
Y = array[:,4]
3.  Use 20% of dataset for testing with the below code:
validation_size = 0.20
3. Each seed value will correspond to a sequence of generated values for a given random number generator:
seed = 7
4.  Training data is in X_train and Y_train for model training and X_validation and Y_validation for later use with the below code:
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
5.  Make predictions on validation dataset.  Adapted from [https://machinelearningmastery.com/machine-learning-in-python-step-by-step/[2]]
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
6. Predict Iris class or species:
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
7. Predict Iris Species (4cm x 4cm sepal and 4cm x 2cm petal)
print(iris.target_names[knn.predict([[3, 5, 3, 3]])])

### Data Analysis:
KNN can be used for predicting class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its nearest K neighbours. It can also be used for regression — output is the value for the object (predicts continuous values). This value is the average (or median) of the values of its k nearest neighbours.  Accuracy is 90% which gives high confidence.  This is a step change in data analytics, machine learning can predict outcomes.
The confusion matrix provides an indication of the three errors made. Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support showing excellent results (granted the validation dataset was small).
KNN can predict an Iris class if data on the 4 variables with the data provided.

## Insights and Comparative Analysis
Data analytics has evolved with the advent of technology and analytical tools that can predict outcomes with high levels of accuracy. However, the Iris dataset can be analysed almost as effectively with only one variable - petal width. Using multiple variables makes simple analysis more difficult to explain and critique.  Also, it has been established that a variety of classification methods yield good results on Iris however Iris is predictable and relatively easy to classify.  

Data analysis on a dataset is only as good as the raw data input.  Versicolor and Virginica are so alike - should additional samples have been taken to further improve and enhance the result instead of applying complex machine learning models to try to predict classification?  It may be that there are issues with the underlying data.  Two of the three samples were picked on the same day - this is a qualitative variable which is not factored into any of the analytical tools.
This project examines statistical, univariate, multivariate and machine learning capabilities.
However the results are only as good as the quality of the data used.

### Conclusion

There are relatively few features in the Iris dataset and two of the classes are much more correlated than the third.  Setosa is very easily separated by the petal width and petal length. Therefore, data modelling lends itself in this case to good results.  We can predict Setosa with almost 100% accuracy given known data points.

In Machine Learning, there is no specific model or an algorithm which can give 100% result to every single dataset. We need to understand the data before we apply any algorithm and build our model depending on the desired result. 

More basic modelling is sufficient as the Iris dataset is predictable.  However, machine learning has undoubted benefits given complexity.

##  References
- [1] Edgar Anderson (1935). "The irises of the Gaspé Peninsula". Bulletin of the American Iris Society. 59: 2–5.
- [2] https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
- [3] https://docs.python.org/3/tutorial/inputoutput.html
- [4] https://www.kaggle.com/uciml/iris
- [5] https://stackoverflow.com/
- [6] https://www.python.org/
- [7] https://www.tutorialspoint.com/
- [8] https://matplotlib.org/
- [9] https://seaborn.pydata.org/
- [10] https://scikit-learn.org/stable/tutorial/basic/tutorial.html
- [11] https://scipy-lectures.org/packages/scikit-learn/index.html
- [12] http://www.learn4master.com/machine-learning/visualize-iris-dataset-using-python
- [13] https://pypi.org/project/tabulate/
- [14] https://bitbucket.org/astanin/python-tabulate
