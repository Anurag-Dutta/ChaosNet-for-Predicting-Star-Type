"""
Created on Thu Mar 02, 2023

Author: Anurag Dutta (anuragdutta.research@gmail.com)
Code Description: A python code to test the efficacy of Decision Tree on the Stars dataset.
Dataset Source: Indian Space Research Organization

"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________
    0.0 (Brown Dwarf)          -     0
    1.0 (Red Dwarf)            -     1
    2.0 (White Dwarf)          -     2
    3.0 (Main Sequence)        -     3
    4.0 (Supergiant)           -     4
    5.0 (Hypergiant)           -     5
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    star            -   Complete Star dataset.    
    X               -   Data attributes.
    y               -   Corresponding labels for X.
    X_train         -   Data attributes for training (80% of the dataset).
    y_train         -   Corresponding labels for X_train.
    X_test          -   Data attributes for testing (20% of the dataset).
    y_test          -   Corresponding labels for X_test.
    X_train_norm    -   Normalizised training data attributes (X_train).
    X_test_norm     -   Normalized testing data attributes (X_test).
_______________________________________________________________________________

ML hyperparameter description:
_______________________________________________________________________________
    MSL     -   The minimum number of samples required to be at a leaf node.
    MD      -   The maximum depth of the tree.
    CCP     -   Complexity parameter used for Minimal Cost-Complexity Pruning. 
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
_______________________________________________________________________________

Performance metric used:
_______________________________________________________________________________
    Macro F1-score (F1SCORE/ f1) -
    The F1 score can be interpreted as a harmonic mean of the precision and 
    recall, where an F1 score reaches its best value at 1 and worst score at 
    0. The relative contribution of precision and recall to the F1 score are 
    equal; 'macro' calculates metrics for each label, and find stheir 
    unweighted mean. This does not take label imbalance into account.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
_______________________________________________________________________________

'''    


#import the star Dataset
star = np.array(pd.read_csv('star.txt', sep="," ,header=None))


#reading data and labels from the dataset
X, y = star[:,range(0,star.shape[1]-1)], star[:,star.shape[1]-1]
y = y.reshape(len(y),1).astype(str)
y = np.char.replace(y, '0.0', '0', count=None)
y = np.char.replace(y, '1.0', '1', count=None)
y = np.char.replace(y, '2.0', '2', count=None)
y = np.char.replace(y, '3.0', '3', count=None)
y = np.char.replace(y, '4.0', '4', count=None)
y = np.char.replace(y, '5.0', '5', count=None)
y = y.astype(int)



#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)


#Normalisation - Column-wise
X_train_norm = (X_train - np.min(X_train,0)) / (np.max(X_train,0) - np.min(X_train,0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = (X_test - np.min(X_test,0)) / (np.max(X_test,0) - np.min(X_test,0))
X_test_norm = X_test_norm.astype(float)



#Algorithm - Decision Tree
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/' 

MSL = np.load(RESULT_PATH+"/h_MSL.npy")[0]
MD = np.load(RESULT_PATH+"/h_MD.npy")[0]
CCP = np.load(RESULT_PATH+"/h_CCP.npy")[0]
F1SCORE = np.load(RESULT_PATH+"/h_F1SCORE.npy")[0]


clf = DecisionTreeClassifier(min_samples_leaf = MSL, random_state = 42, max_depth = MD, ccp_alpha = CCP)
clf.fit(X_train_norm, y_train.ravel())


y_pred = clf.predict(X_test_norm)
f1 = f1_score(y_test, y_pred, average='macro')


print('TRAINING F1 Score', F1SCORE)
print('TESTING F1 Score', f1)

np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )

