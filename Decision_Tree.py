from random import randint

import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from DataPreparing import *

if __name__ == "__main__":
    object_file_name = "Data/OWASP_Trainable_Dataset.pkl"

    # separating features and labels
    X_data, Y_data = load_data(object_file_name)

    # print(X_data.shape, Y_data.shape)

    # default criterion = gini
    """
    max_depth: The max_depth parameter denotes maximum depth of the tree. It can take any integer value or None. 
    If None, then nodes are expanded until all leaves are pure or until all leaves contain less than 
    min_samples_split samples. By default, it takes “None” value. This will often result in over-fitted decision 
    trees. The depth parameter is one of the ways in which we can regularize the tree, or limit the way it grows to 
    prevent over-fitting. 

    min_samples_leaf: The minimum number of samples required to be at a leaf node.
    If an integer value is taken then consider min_samples_leaf as the minimum no.
    If float, then it shows percentage. By default, it takes “1” value. 

    """
    f = open("gini_result.txt", "w")
    # clf_gini = DecisionTreeClassifier(random_state=None, min_samples_leaf=5)

    param_dist = {"max_depth": list(range(1, 9)),
                  "max_features": list(range(1, 35)),
                  "min_samples_leaf": list(range(1, 18)),
                  "criterion": ["gini", "entropy"]}

    # Instantiate a Decision Tree classifier: tree
    tree = DecisionTreeClassifier()

    # Instantiate the RandomizedSearchCV object: tree_cv
    kf = KFold(n_splits=10, random_state=0, shuffle=True)
    tree_cv = GridSearchCV(tree, param_dist, cv=kf)

    # Fit it to the data
    tree_cv.fit(X_data, Y_data)
    # Print the tuned parameters and score
    print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
    print("Best score is {}".format(tree_cv.best_score_))
