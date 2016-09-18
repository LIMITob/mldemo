# -*- coding: utf-8 -*-
"""
Thie file implements a simple RandomForestClassifier
"""
__auther__ = 'ZT.Chow'

import pandas as pd
import math

from preprocessing import pre_process
from collections import Counter


class Node(object):
    def __init__(self,father,data):
        self.father = father
        self.data = data
        self.child = {}
        self.result = None
    def set_feature(self,feature):
        self.feature = feature
    def set_result(self,result):
        self.result = result
    def set_value(self,value):
        self.value = value
        
class DecisionTree(object):
    def __init__(self,features,dataset,limit=20):
        self.dataset = dataset
        self.features = []
        #self.dataset = dataset
        self.root = Node(None,dataset)
        
    def buildTree(self,root,dataset,features,value=None):
        
        current_node = Node(root,dataset)
        
        if None != root:
            root.child[value] = current_node
            
        current_data = dataset
        current_features = features
        
        # the data has been split to a threshold
        if len(dataset) > self.limit or (current_features) > 1:
            
            current_feature = feature_selection(current_data,current_features)
            current_node.set_feature(current_feature)
            
            for i in list(Counter(titanic[current_feature])):
                child_data = [current_data[current_feature]==i]
                self.buildTree(current_node, child_data, features, i)
        else:
            result = list(Counter(current_data['Survived']).most_common(1))[0]
            current_node.set_result(result)
            current_node.child = None
        return
    
    def search(self,data, root=self.root):
        current = root
        if current.child == None:
            return current.result
        else:
            self.search(current.child[data[current.feature]])
        
    def classify(self, dataset):
        result = []
        for item in dataset:
            result.append(self.search(item))
        return result
        
class RandomForestClassifier(object):
    
    def __init__(self,):
        self.tree_num = 10

        pass

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
        entropy += tmp_prob*math.log(tmp_prob,2)
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
    titanic = titanic[titanic['Embarked']=='Q']
    print titanic[:10]
    features = list(titanic.columns)[1:]
    print feature_selection(titanic,features)