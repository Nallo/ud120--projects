#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

### import the sklearn module for SVMs
from sklearn.svm import SVC

### create classifier
clf = SVC(kernel="rbf", C=10000.)

### One way to speed up an algorithm is to train it on a smaller training dataset.
### The tradeoff is that the accuracy almost always goes down when you do this.
### reduce the training data to 1%
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
print "number of features: ", len(features_train[0])

### fit the classifier on the training features and labels
### and compute the time to fit data
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

### use the trained classifier to predict labels for the test features
### and compute the time to predict the features_test
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

print "Item #10", pred[10]
print "Item #26", pred[26]
print "Item #50", pred[50]

import numpy as np
print "Element in class 1:", np.count_nonzero(pred)

### calculate and return the accuracy on the test data
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)

### display the accuracy
print "Accuracy:", round(accuracy, 3)


#########################################################
