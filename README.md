# Artificial-Intelligence
## 1. Clickstream Mining with Decision Trees:
The project is based on a task posed in KDD Cup 2000. It involves mining click-stream data collected from Gazelle.com, which sells legware products.Given a set of page views, will the visitor view another page on the site or will he leave?
Implemented the ID3 decision tree learner on Python.

DATASET:
5 files in .csv format:
trainfeat.csv: Contains 40000 examples, each with 274 features in the form of a 40000 x 274 matrix.
trainlabs.csv: Contains the labels (class) for each training example (did the visitor view another page?)
testfeat.csv: Contains 25000 examples, each with 274 features in the form of a 25000 x 274 matrix.
testlabs.csv: Contains the labels (class) for each testing example.
featnames.csv: Contains the "names" of the features. These might useful if you need to know what the features represent.
The format of the files is not complicated, just rows of integers separated by empty spaces.

Command to run in python:
python q1_classifier.py -p <pvalue> -f1 <train_dataset> -f2 <test_dataset> -o <output_file> -t <decision_tree>
-p to set p-value threshold for the chi-square stopping criteria 
-f1 to load the training data .csv file
-f2 to load the testing data .csv file
-o to generate the csv file for output
-t to store the decision tree generated 


## 2. Spam Filter :

The dataset we will be using is a subset of 2005 TREC Public Spam Corpus. ImplementED the Naive Bayes algorithm to classify spam.

DATASET:
-It contains a training set and a test set. 
-Both files use the same format: each line represents the space-delimited properties of an email, with the first one being the email ID, the second one being whether it is a spam or ham (non-spam), and the rest are words and their occurrence numbers in this email. In preprocessing, non-word characters have been removed, and features selected similar to what Mehran Sahami did in his original paper using Naive Bayes to classify spams.

Command to run in python:
 python q2_classifier.py -f1 <train_dataset> -f2 <test_dataset> -o <output_file>
-f1 to load the training data .csv file
-f2 to load the testing data .csv file
-o to generate the csv file for output
