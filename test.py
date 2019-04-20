# labelnames translates the species names into integers
labelnames = {"Iris-setosa\n":0, 
              "Iris-versicolor\n":1, 
              "Iris-virginica\n":2}
data, labels = [], []      # Empty lists for the vectors, labels.
f = open("iris.data", "r") # Open the data file.
for line in f:             # Load in each line from the file. 
    vals = line.split(',') # Split the line into individual values
    if len(vals) == 5:     # Check that there are five columns
        data.append(vals[:-1])              # Add data vector
        labels.append(labelnames[vals[4]])  # Add numerical label
f.close()                  # Close the file

import sklearn.linear_model

# Create the Logistic Regression object
LR = sklearn.linear_model.LogisticRegression()

LR.fit(data[50:], labels[50:]) # Find the hyperplane
pred = LR.predict(data[50:])   # Predict the classes

# Count the number of correct predictions
correct = len([i for i in range(100) if pred[i] == labels[i+50]])