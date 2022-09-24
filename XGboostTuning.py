import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.svm import SVC
import xgboost as xgb
from DataPreparing import load_data
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

def k_fold_dart(X,Y,clf, num_round):
    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    kf.get_n_splits(X)

    accuracies = []
    recalls = []
    precisions = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test,num_round)
        Y_pred = [round(value) for value in Y_pred]
        accuracies.append(metrics.accuracy_score(Y_test, Y_pred))
        recalls.append(metrics.recall_score(Y_test, Y_pred))
        precisions.append(metrics.precision_score(Y_test, Y_pred))

    average_accuracy = np.mean(np.array(accuracies))
    average_recall = np.mean(np.array(recalls))
    average_precision = np.mean(np.array(precisions))
    return average_accuracy, average_recall, average_precision


if __name__ == "__main__":
    object_filename = "/kaggle/input/input-files/OWASP_Trainable_Dataset.pkl"
    X,Y = load_data(object_filename)
    print(X.shape, Y.shape)
    
    #hyper parameters for gblinear
    boosters = ["gbtree", "dart"]
    objective = ["reg:squarederror","binary:logistic","reg:logistic"]
    learningRate = [0.001, 0.01, 0.1, 0.7]
    depth = np.arange(3,7,1)
    minChildWeight = [0.001, 0.01,0.1, 1, 5, 10, 50]
    subsample = [0.7,0.8,1.0]
    colSamplebyTree = [0.8, 1.0]
    lmda = [0.05, 0.1, 1, 5]
    gamma = [0, 0.5, 1, 2]
    
    # boosters = ["gbtree", "dart"]
    # #objective = ["reg:squarederror","binary:logistic","reg:logistic"]
    # objective = ["reg:logistic"]
    # learningRate = [0.05, 0.3, 0.7]
    # depth = [6,7,8]
    # minChildWeight = [0.01,0.1, 1, 10, 50]
    # subsample = [0.7,0.8,1.0]
    # colSamplebyTree = [0.8, 1.0]
    # lmda = [ 1 ]
    # gamma = [ 2 ]
    
    #files
    gbtreeFile = open("gbtree13.csv","w")
    fieldnames = ['accuracy','precision','recall','objective','learning_rate','max_depth','min_child_weight', 'subsample', 'colSamplebyTree', 'lambda', 'gamma']
    gbwriter = csv.DictWriter(gbtreeFile, fieldnames=fieldnames)
    gbwriter.writeheader()
    
    dartFile = open("dart13.csv","w")
    dartwriter = csv.DictWriter(dartFile, fieldnames=fieldnames)
    dartwriter.writeheader()
    
    
    resultNo = 0
    for b in boosters:
        for obj in objective:
                for l in learningRate:
                    for d in depth:
                        for mcw in minChildWeight:
                            for s in subsample:
                                for cs in colSamplebyTree:
                                    for lm in lmda:
                                        for g in gamma:
                                            clf = xgb.XGBRegressor(booster=b, objective=obj,learning_rate=l,max_depth=d,min_child_weight=mcw,subsample=s,colsample_bytree=cs,reg_lambda=lm,gamma=g,random_state=42)
                                            if b == "gbtree":
                                                try:
                                                    average_accuracy, average_recall, average_precision = k_fold(X, Y, clf)
                                                except Exception as e:
                                                    print(e)
                                                    print(str(b)+","+str(obj))
                                                    continue
                                                gbwriter.writerow({"accuracy":average_accuracy*100.0,"precision":average_precision*100,"recall":average_recall*100,"objective":obj,"learning_rate":l,
                                                               "max_depth":d,"min_child_weight":mcw, "subsample":s,"colSamplebyTree":cs,"lambda":lm,"gamma":g})
                                            
                                                print("iteration No: "+str(resultNo)+"\n")
                                                #resultNo = resultNo + 1
                                            else:
                                                try:
                                                    average_accuracy, average_recall, average_precision = k_fold_dart(X, Y, clf, 50)
                                                except Exception as e:
                                                    print(e)
                                                    print(str(b)+","+str(obj))
                                                    continue
                                                dartwriter.writerow({"accuracy":average_accuracy*100.0,"precision":average_precision*100,"recall":average_recall*100,"objective":obj,"learning_rate":l,
                                                               "max_depth":d,"min_child_weight":mcw, "subsample":s,"colSamplebyTree":cs,"lambda":lm,"gamma":g})
                                            
                                                print("iteration No: "+str(resultNo)+"\n")
                                            resultNo = resultNo + 1
                                                
            
                
            
    
    gbtreeFile.close()
    dartFile.close()
