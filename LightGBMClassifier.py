import pickle

import lightgbm
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier
import csv
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold, RandomizedSearchCV
import xgboost as xgb


def load_data(filename):
    """

    :param filename:
    :return:
    """
    file = open(filename, 'rb')
    obj = pickle.load(file)
    file.close()
    X = obj["features"]
    Y = obj["labels"]
    return X, Y


def k_fold(X, Y, clf):
    #  shuffle dataset randomly , split the dataset into 10 groups
    kf = KFold(n_splits=10, random_state=1, shuffle=True)

    accuracies = []
    recalls = []
    precisions = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        Y_pred = [round(value) for value in Y_pred]
        ac = metrics.accuracy_score(Y_test, Y_pred)
        rc = metrics.recall_score(Y_test, Y_pred)
        pr = metrics.precision_score(Y_test, Y_pred)
        accuracies.append(ac)
        recalls.append(rc)
        precisions.append(pr)
    average_accuracy = np.mean(np.array(accuracies))
    average_recall = np.mean(np.array(recalls))
    average_precision = np.mean(np.array(precisions))
    return average_accuracy, average_recall, average_precision


if __name__ == "__main__":
    object_file_name = "/kaggle/input/owasp-trainable-datasetpkl/OWASP_Trainable_Dataset.pkl"

    # separating features and labels
    X_data, Y_data = load_data(object_file_name)

    # regression list
    fieldnames = ['accuracy', 'precision', 'recall', 'objective', 'boosting_type', 'n_estimators', 'learning_rate', 'max_depth', 'num_leaves',
                  'min_child_samples', 'max_bin', 'subsample', 'subsample_freq', 'colsample_bytree', 'reg_lambda', 'reg_alpha']
    objective_list = ['binary']
    booster_list = ['dart', 'gbdt']
    max_depth_list = [6, 7]  # 3
    num_leaves_list = list(range(40, 80, 8))  # 5
    min_child_samples_list = [50 , 100]
    min_child_weight_list = [10, 20, 30]  # 3
    n_estimators = [380]

    max_bin_list = [100, 255]  # 2
    subsample_list = [0.7, 1.0]  # 2
    colsample_bytree_list = [0.7, 1.0]  # 2
    learning_rate_list = [0.005, 0.01, 0.05]  # 3
    reg_lambda_list = [0.0, 1.2, 1.4]  # 3
    reg_alpha_list = [1.0, 1.2]  # 2

    resultNo = 0
    for objective in objective_list:
        for booster in booster_list:
            filePath = "LightGBMClassifier" + objective + "_" + booster + "_result.csv"
            file = open(filePath, "w")
            resultWriter = csv.DictWriter(file, fieldnames=fieldnames)
            resultWriter.writeheader()
            for max_depth in max_depth_list:
                for num_leaves in num_leaves_list:
                    for learning_rate in learning_rate_list:
                        for min_child_samples in min_child_samples_list:
                            for max_bin in max_bin_list:
                                for subsample in subsample_list:
                                    for colsample_bytree in colsample_bytree_list:
                                        for reg_lambda in reg_lambda_list:
                                            for reg_alpha in reg_alpha_list:
                                                clf = lightgbm.LGBMClassifier(
                                                    objective=objective, boosting_type=booster, max_depth=max_depth,
                                                    n_estimators=380,
                                                    num_leaves=num_leaves,
                                                    max_bin=max_bin,
                                                    subsample=subsample,
                                                    colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                                                    reg_alpha=reg_alpha,
                                                    learning_rate=learning_rate, subsample_freq=5)

                                                average_accuracy, average_recall, average_precision = k_fold(X_data,
                                                                                                             Y_data,
                                                                                                             clf)

                                                resultWriter.writerow(
                                                    {"accuracy": average_accuracy * 100,
                                                     "precision": average_precision * 100,
                                                     "recall": average_recall * 100,
                                                     'objective': objective, 'boosting_type': booster,
                                                     'n_estimators':380,
                                                     'max_depth': max_depth,
                                                     'num_leaves': num_leaves,
                                                     'min_child_samples': min_child_samples, 'max_bin': max_bin,
                                                     'subsample': subsample,
                                                     'subsample_freq': 5,
                                                     'colsample_bytree': colsample_bytree, 'reg_lambda': reg_lambda,
                                                     'reg_alpha': reg_alpha,
                                                     'learning_rate': learning_rate,
                                                     })

                                                print("iteration No: " + str(resultNo) + "\n")
                                                resultNo += 1
            file.close()
