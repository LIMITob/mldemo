# -*- coding: utf-8 -*-
"""
Thie file implements a simple RandomForestClassifier
"""
__auther__ = 'ZT.Chow'

__all__ = ['Node',
           'DecisionTree',
           'RandomForestClassifier']
        
import pandas as pd
import numpy as np
import random
import math

from preprocessing import pre_process
from collections import Counter


class Node(object):
    def __init__(self,father,data):
        self.father = father
        self.data = data
        self.child = {}
        self.result = -1
        self.feature = None
        self.value = None
        
    def set_feature(self, feature):
        self.feature = feature
    def set_result(self, result):
        self.result = result
        
class DecisionTree(object):
    def __init__(self,dataset,limit=0):
        self.dataset = dataset
        self.features = list(dataset.columns)
        del self.features[self.features.index('Survived')]
        self.limit = limit
        #self.dataset = dataset
        self.root = None
        self.result = []

    def buildTree(self,root,dataset,features,value=None):
        #print len(dataset)
        if None != root:
            current_node = Node(root,dataset)
            root.child[value] = current_node
        else:
            current_node = Node(None,dataset)
            self.root = current_node
            
        current_data = dataset
        current_features = features
        
        # the data has been split to a threshold
        if len(current_data) > self.limit and len(current_features) > 1:
            current_feature = feature_selection(current_data,current_features)
            current_node.set_feature(current_feature)
            #create the children node 
            for i in list(Counter(titanic[current_feature])):
                child_data = current_data[current_data[current_feature]==i]
                index = current_features.index(current_feature)
                cfeatures = current_features[0:index] + current_features[index+1:]
                
                if len(child_data) > 0:
                    self.buildTree(current_node, child_data, cfeatures, i) 
        #`do some stuff when the node is leafnode
        else:
            result = Counter(current_data['Survived'])
            #print result.most_common(1)[0][0]
            current_node.set_result(result.most_common(1)[0][0])
            current_node.child = None
        return
    
    def search(self, data, root):
        current = root
        if current.child == None:
            #print '------------------------',current.result
            return current.result
        else:
            #print current.feature,'|',list(data[current.feature])[0],'|',current.result
            print current.feature, len(current.child) ,list(data[current.feature])
            return self.search(data,current.child[list(data[current.feature])[0]])
            
    def classify(self, dataset ,root):
        result = []
        for i in range(len(dataset)):
            result.append(self.search(dataset[i:i+1], root))
        return result
        
    def score(self,x,y):
        result = self.classify(x,self.root)
        #print result,type(y)
        return Counter(list(result == y))[1]/float(len(x))
                
        
class RandomForestClassifier(object):
    def __init__(self,tree_num=1,data_num=0.6,freature_num=0.6,criterion='infogain'):
        self.tree_num = tree_num
        self.data_num = data_num
        self.feature_num = freature_num
        self.trees = []
        
    def train(self,X,y):
        features = list(X.columns)
        for i in range(self.tree_num):
            dindex = np.random.randint(0,len(X),(int(self.data_num*len(X)),))          
            findex = random.sample(features[1:], int((1-self.feature_num)*len(features)))            
            dcurrent = X.loc[dindex]
            #print features-findex
            for j in findex:                
                del dcurrent[j]
            t = DecisionTree(dcurrent)
            t.buildTree(t.root,t.dataset,t.features)
            self.trees.append(t)
        if len(self.trees) == self.tree_num:
            print 'Build Success!'
        return 
        
    def classify(self,X):
        result = []
        for i in self.trees:
            result.append(i.classify(X,i.root))
        return result
        
    def score(self,X,y):
        return
#Type(dataset) == pd.DataFrame 
def calc_entropy(dataset):
    num_entries = len(dataset)
    entropy = 0.0
    label_counts = {}
    for item in dataset['Survived']:
        label_counts.setdefault(item,0)
        label_counts[item] += 1
        
    for key in label_counts:
        tmp_prob = float(label_counts[key])/num_entries
        # when 0log0 ,let it equals zero
        if tmp_prob == 0:
            continue
        entropy += tmp_prob * math.log(tmp_prob, 2)
    return 0 - entropy
 
# Type(dataset) == pd.DataFrame 
def calc_cond_entropy(dataset,cond_item):
    num_entries = len(dataset)
    cond_item_counts = {}
    for item in dataset[cond_item]: 
        cond_item_counts.setdefault(item,0)          
        cond_item_counts[item] += 1

    cond_split_datset = {}
    for key in cond_item_counts:
        cond_split_datset[key] = dataset[dataset[cond_item]==key]

    cond_entropy = 0.0
    for key in cond_item_counts:
        tmp_prob = float(cond_item_counts[key])/num_entries
        if tmp_prob == 0:            
            continue
        cond_entropy += tmp_prob*calc_entropy(cond_split_datset[key])
    return cond_entropy

def feature_selection(dataset,features):
    entropy = calc_entropy(dataset)
    factor = []
    for i in features:
        factor.append(entropy - calc_cond_entropy(dataset,i))
        #print entropy - calc_cond_entropy(dataset,i),
    index = factor.index(max(factor))
    return features[index]
    
if __name__=='__main__':
    titanic,test = pre_process()
    #print titanic[:10]
    t = DecisionTree(titanic)
    t.buildTree(t.root,t.dataset,t.features)
    rf = RandomForestClassifier()
    rf.train(titanic,titanic['Survived'])
    #result = rf.classify(titanic)
    print len(rf.trees)
    print t.score(titanic,titanic['Survived'])
    