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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel="rbf", C=10000, gamma="auto")

t0 = time()
# 1% of training set
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

# training the algorithm
clf.fit(features_train, labels_train)
# print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
# print "predicting time:", round(time()-t0, 3), "s"

## finding accuracy
# accuracy = accuracy_score(pred, labels_test)
# print "accuracy:", accuracy

## predicting 10, 26, 50th elements in test set
# author_10 = pred[10]
# print "author of email 10:", author_10

# author_26 = pred[26]
# print "author of email 26:", author_26

# author_50 = pred[50]
# print "author of email 50:", author_

# How many emails from "Chris" are expected?
print sum(pred)
#########################################################


