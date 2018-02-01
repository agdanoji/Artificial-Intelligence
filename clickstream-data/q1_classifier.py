from __future__ import division
import pandas as pd
import math
import numpy as np

train_examples = []
train_feature_vector = []
train_labels_vector = []
test_feature_vector = []
test_labels_vector = []

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take

    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Does not matter, you can leave them the same or cast to None.

'''


# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes = list(children)
        self.data = data

def getFeatureVectors(train_examples):
    n_Features = len(train_examples[0])
    n_Examples = len(train_examples)
    distinctfeatures =[]
    fullfeature=[]
    for feature in range (n_Features):
        v_distinct =[]
        v_fullfeature = []
        for example in range(n_Examples):
            v_fullfeature.append(train_examples[example][feature])
            if train_examples[example][feature] not in v_distinct:
                v_distinct.append(train_examples[example][feature])
        distinctfeatures.append(v_distinct)
        fullfeature.append(v_fullfeature)

    return distinctfeatures,fullfeature



def parseData():
    # data_featureNames=pd.read_csv("C:/Users/Himani/Desktop/AI/clickstream-data.tar/clickstream-data/featnames.csv",header=None)
    data_trainFeatures = pd.read_csv("train.csv",header=None)
    data_trainLabels = pd.read_csv("train_label.csv", header=None)
    data_testFeatures=pd.read_csv("test.csv",header=None)
    data_testLabels=pd.read_csv("test_label.csv",header=None)



    # dataframe to list conversion for given examples where each row represents one example.
    temp_train_data = data_trainFeatures.values.T.tolist()

    # train features
    for val in temp_train_data[0]:
        res = val.split(' ')
        for i in range(len(res)):
            res[i] = int(res[i])
        train_examples.append(res)

    # column wise splitting to get list of feature vectors with distinct values it can take
    train_labels_vector=data_trainLabels
    distinctFeatures, fullFeatures = getFeatureVectors(train_examples)


    # dataframe to list conversion for trained features.
    for index, row in data_trainLabels.T.iterrows():
        train_labels_vector = row.values
    #print(train_labels_vector)

    Houtput= entropy(train_labels_vector)
    #print Houtput
    #print('df', len(distinctFeatures))

    list_entropy=[]
    ig=[]
    for k in range(len(train_examples[0])):
        #print k
        entropy_feature=0
        for i in distinctFeatures[k]:
             label=[]
             for j in range(len(fullFeatures[k])):
                if i is fullFeatures[k][j]:
                    label.append(train_labels_vector[j-1])
             entropy_for_each_featval= entropy(label)
             entropy_feature+= float(len(label)/len(train_labels_vector))*entropy_for_each_featval
        list_entropy.append(entropy_feature)
    #print list_entropy

    for i in list_entropy:
        ig.append(Houtput-i)
    #print ig

    return ig

#def create_decision_tree(node):


def entropy(labels):

    n_labels = len(labels)

    if n_labels <=1:
        return 0

    counts = np.bincount(labels)

    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)

def main():
    ig=parseData()
    print ig
    print max(ig)
    print ig.index(max(ig))
    root= TreeNode()
    attr_boll=[]
    for i in range(len(train_examples[0])):
        attr_boll.append(0)


if __name__ == "__main__":
    main()

