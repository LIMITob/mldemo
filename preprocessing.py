# -*- coding: utf-8 -*-
"""
Thie file did the preprocessing for dataset titanic event
"""
__auther__ = 'ZT.Chow'

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('whitegrid')

def pre_process():
    titanic = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')
    
    #get rid of some useless info which do nothing to predict
    titanic = titanic.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
    test = test.drop(['Name','Ticket','Cabin'],axis=1)
    
    # fill the blanket of Embarked with S 
    titanic['Embarked'] = titanic['Embarked'].fillna('S')
    
    # fill the loss data of Age with the median
    titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())   
    test['Age'] = test['Age'].fillna(test['Age'].median())
    
    # Let the Age ensumble to 3 level
    titanic.loc[titanic['Age'] <=20,'Age'] = 0
    titanic.loc[titanic['Age'] >50,'Age'] = 2
    titanic.loc[(titanic['Age']>20)&(titanic['Age'] <=50),'Age'] = 1
    
    # let the Fare ensemble to 3 level
    titanic.loc[titanic['Fare'] <=10,'Fare'] = 0
    titanic.loc[(titanic['Fare']>10)&(titanic['Fare'] <=50),'Fare'] = 1
    titanic.loc[(titanic['Fare']>50)&(titanic['Fare'] <=100),'Fare'] = 2
    titanic.loc[titanic['Fare'] >100,'Fare'] = 3
    
    return titanic,test


if __name__=='__main__':
    titanic,test = pre_process()
    
    #print titanic[:10],list(titanic.columns)
    #print titanic['Fare'].value_counts()
    rarray = [1,2,3,4,5,6,7,8,9,10]
    print np.random.sample(rarray).shape
