from DataPreparing import load_data
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.svm import SVC
import xgboost as xgb
import csv
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor


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

#clf=ExtraTreesClassifier(n_estimators = 200, criterion ='entropy', max_features = 30)

    extratreeFile = open("ExtraTree/extra.csv","a", newline='')
    fieldnames = ['accuracy','precision','recall', 'estimators','depth','max_features','criterion']
    lrwriter = csv.DictWriter(extratreeFile, fieldnames=fieldnames)
    lrwriter.writeheader()


n_estimators = np.linspace(10, 400, 40, endpoint=True, dtype=int)
depth = np.linspace(2, 32, 16, endpoint=True, dtype=int)
max_features = ["auto","sqrt","log2"]
criterion = ["mse","mae"]
type = ["r1","c1"]
min_samples_split = np.linspace(5, 30, 6, endpoint=True, dtype=int)
min_samples_leaf= np.linspace(5, 50, 10, endpoint=True, dtype=int)

count=0
for h in type:
    for i in criterion:
       for j in max_features:
            for k in depth:
               for l in n_estimators: 
                   for m in min_samples_split:
                       for n in min_samples_leaf:
                           if h != "c1" :
                            clf=ExtraTreesRegressor(n_estimators=l, criterion=i, max_features=j, max_depth=k,n_jobs=-1, min_samples_split=m,min_samples_leaf=25)
                   average_accuracy, average_recall, average_precision = k_fold(X, Y, clf)
                   lrwriter.writerow({"accuracy":average_accuracy*100, "precision":average_precision*100, "recall": average_recall*100
                   ,"estimators": l, "depth": k ,"max_features": j, "criterion": i})
                   if h != "r1" :                                 
                      clf=ExtraTreesRegressor(n_estimators=l ,max_features=j, max_depth=k,n_jobs=-1, min_samples_split=m,min_samples_leaf=n)
                      average_accuracy, average_recall, average_precision = k_fold(X, Y, clf)
                      lrwriter.writerow({"accuracy":average_accuracy*100, "precision":average_precision*100, "recall": average_recall*100
                     ,"estimators": l, "depth": k ,"max_features": j, "criterion": 'None'})
                  
                   count = count +1;
                   print(count)

extratreeFile.close()




