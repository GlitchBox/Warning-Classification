from DataPreparing import load_data
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.svm import SVC
import xgboost as xgb


def k_fold(X, Y, clf):
    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    kf.get_n_splits(X)

    accuracies = []
    recalls = []
    precisions = []

    print(kf.split(X))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        # print(train_index)
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
    X, Y = load_data(object_filename)
    print(X.shape, Y.shape)

    clf = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    average_accuracy, average_recall, average_precision = k_fold(X, Y, clf)
    print("Accuracy:", average_accuracy)
    print("Precision:", average_precision)
    print("Recall:", average_recall)
