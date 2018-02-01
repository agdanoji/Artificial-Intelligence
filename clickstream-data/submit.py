import random
import pickle as pkl
import argparse
import csv
import numpy as np
from collections import Counter

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take

    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''


# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self, filename):
        obj = open(filename, 'w')
        pkl.dump(self, obj)


# loads Train and Test data
def load_data(ftrain, ftest):
    Xtrain, Ytrain, Xtest = [], [], []
    with open(ftrain, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            Xtrain.append(rw)

    with open(ftest, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            Xtest.append(rw)

    ftrain_label = ftrain.split('.')[0] + '_label.csv'
    with open(ftrain_label, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = int(row[0])
            Ytrain.append(rw)

    print('Data Loading: done')
    return Xtrain, Ytrain, Xtest


num_feats = 274


# # A random tree construction for illustration, do not use this in your code!
# def create_random_tree(depth):
#     if(depth >= 7):
#         if(random.randint(0,1)==0):
#             return TreeNode('T',[])
#         else:
#             return TreeNode('F',[])

#     feat = random.randint(0,273)
#     root = TreeNode(data=str(feat))

#     for i in range(5):
#         root.nodes[i] = create_random_tree(depth+1)

# return root

# def entropy(labels):

#     n_labels = len(labels)

#     if n_labels <=1:
#         return 0

#     counts = np.bincount(labels)

#     probs = counts[np.nonzero(counts)] / n_labels
#     n_classes = len(probs)

#     if n_classes <= 1:
#         return 0
#     return - np.sum(probs * np.log(probs)) / np.log(n_classes)

def calculateEntropy(labels):
    labelCounts = Counter(labels)
    # print labelCounts
    totalRows = len(labels)
    # print totalRows
    e = 0
    for l, numL in labelCounts.iteritems():
        # print 'l', l
        # print 'numL', numL
        # print 'totalRows', totalRows
        x = float(numL) / totalRows
        # print 'x', x
        e = e + (-1 * x * np.log10(x))
    # print 'entropy', e

    return e


def calculateGain(featureValues, labels, currentEntropy):
    uniqueFeatureCounts = Counter(featureValues)
    totalRows = len(labels)
    featureEntropy = 0

    if (len(uniqueFeatureCounts) <= 1):
        return 0
    else:
        for u, numU in uniqueFeatureCounts.iteritems():
            thisLabels = []
            index = 0
            for f in featureValues:
                if (f == u):
                    thisLabels.append(labels[index])
                index = index + 1
            featureEntropy = featureEntropy + ((float(numU) / totalRows) * calculateEntropy(thisLabels))

    return currentEntropy - featureEntropy


def splitDecider(allData, currentRows, unUsedFeatures, dataLabel):
    # print 'yy'
    # print unUsedFeatures
    data = []
    for r in currentRows:
        data.append(dataLabel[r])

    currentEntropy = calculateEntropy(data)
    gainList = []

    for f in unUsedFeatures:
        featureValues = []
        labels = []
        for r in currentRows:
            labels.append(dataLabel[r])
            featureValues.append(allData[r][f])
            #print featureValues
        gainList.append(calculateGain(featureValues, labels, currentEntropy))

    maxIndex = gainList.index(max(gainList))
    print max(gainList)
    # print maxIndex
    returnValue = unUsedFeatures[maxIndex]
    # print 'aaaaaaaaaaaaaa'
    unUsedFeatures.remove(returnValue)
    # print unUsedFeatures
    # print 'aaaaaaaaaaaaaa'
    return returnValue, unUsedFeatures


def chiStop():
    return 0


def make_tree(allData, currentRows, unUsedFeatures, dataLabel):
    if currentRows:
        data = []
        for r in currentRows:
            data.append(dataLabel[r])
        if (calculateEntropy(data) == 0):
            return TreeNode(dataLabel[0], [])
        elif (chiStop() == 1):
            return TreeNode('T', [])
        else:
            splitFeature, unUsedFeatures = splitDecider(allData, currentRows, unUsedFeatures, dataLabel)
            # print 'ppppppppppp'
            # print unUsedFeatures
            # print 'ppppppppppp'
            root = TreeNode(data=splitFeature)
            newRow = [[] for x in range(len(Counter(allData[:][splitFeature])))]
            for r in currentRows:
                newRow[allData[r][splitFeature] - 1].append(r)
            for i in newRow:
                root.nodes[i] = make_tree(allData, i, unUsedFeatures, dataLabel)
            return root


parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[
                  0] + '_labels.csv'  # labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']

# t = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
# h = [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
# c = 0.94
# print calculateGain(h, t, c)
# print calculateEntropy([1,1,1,1,1,1,1,1])

Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)
# print len(Xtrain[1])
# print len(Ytrain)

# x = [2, 3, 5]
# splitDecider(Xtrain, x, Xtrain)
# print x
currentRows = []
for i in range(len(Xtrain)):
    currentRows.append(i)

unUsedFeatures = []
for i in range(len(Xtrain[0])):
    unUsedFeatures.append(i)

print("Training...")
# s = create_random_tree(0)
s = make_tree(Xtrain, currentRows, unUsedFeatures, Ytrain)
s.save_tree(tree_name)
print("Testing...")
Ypredict = []
# generate random labels
for i in range(0, len(Xtest)):
    Ypredict.append([np.random.randint(0, 2)])

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")








