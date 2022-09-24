# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 23:48:32 2019

@author: Asus
"""

from DataPreparing import load_data
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.svm import SVC
import xgboost as xgb
import csv


def k_fold(X, Y, clf):
    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    kf.get_n_splits(X)

    accuracies = []
    recalls = []
    precisions = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        Y_pred = [round(value) for value in Y_pred]
        accuracies.append(metrics.accuracy_score(Y_test, Y_pred))
        recalls.append(metrics.recall_score(Y_test, Y_pred))
        precisions.append(metrics.precision_score(Y_test, Y_pred))

    average_accuracy = np.mean(np.array(accuracies))
    average_recall = np.mean(np.array(recalls))
    average_precision = np.mean(np.array(precisions))
    return average_accuracy, average_recall, average_precision


if __name__ == "__main__":
    object_filename = "Data/OWASP_Trainable_Dataset.pkl"
    X,Y = load_data(object_filename)
    print(X.shape, Y.shape)
    
    lrFile = open("LRResults/lr.csv","a")
    fieldnames = ['accuracy','precision','recall','solver','C', 'penalty']
    lrwriter = csv.DictWriter(lrFile, fieldnames=fieldnames)
    lrwriter.writeheader()
    
    #hyper parameters
    cVal = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,100]
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    #max_iter = np.arange(50,500, 50)
    penalty = ['l1','l2']
    
    resultNo = 0
    for s in solver:
        for c in cVal:
            for p in penalty:
                 
                mi = 0
                if s=="lbfgs":
                    mi = 1000
                else:
                    mi = 100
                    
                if s != "liblinear" and s!="saga":
                    if p=='l2':
                        clf = LogisticRegression(solver=s, penalty = p, C = c, max_iter=10000)
                        average_accuracy, average_recall, average_precision = k_fold(X, Y, clf)
                        lrwriter.writerow({"accuracy":average_accuracy*100, "precision":average_precision*100, "recall": average_recall*100,
                                        "solver":s, "C":c, "penalty":p})
                else:
                    clf = LogisticRegression(solver=s, penalty = p, C = c, max_iter = 10000)
                    average_accuracy, average_recall, average_precision = k_fold(X, Y, clf)
                    lrwriter.writerow({"accuracy":average_accuracy*100, "precision":average_precision*100, "recall": average_recall*100,
                                        "solver":s, "C":c, "penalty":p}) 

                                
                print("iteration No: "+str(resultNo)+"\n")
                resultNo = resultNo + 1
    lrFile.close()

            
    
