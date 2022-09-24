from DataPreparing import load_data
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import csv


def featureRanking(fileName, decisionMetric):
    object_filename = "Data/OWASP_Trainable_Refined_Dataset.pkl"
    X, Y, cols = load_data(object_filename)
    # print("Shape:")
    # print(X.shape, Y.shape)
    # print("cols:")
    # print(cols)

    extra_tree_forest = ExtraTreesClassifier(n_estimators = 20, 
                       criterion =decisionMetric, max_features = 5) 
    #extra_tree_forest = ExtraTreesClassifier()

    # Training the model 
    extra_tree_forest.fit(X, Y) 

    # Computing the importance of each feature 
    feature_importance = extra_tree_forest.feature_importances_ 

    # Normalizing the individual importances 
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                            extra_tree_forest.estimators_], 
                                            axis = 0)

    ranking = []
    for i in range(len(cols)):
        rankingInfo = {"importance":feature_importance_normalized[i], "feature":cols[i]}
        ranking.append(rankingInfo)

    ranking = sorted(ranking, key = lambda p: (p['importance']), reverse =True)

    with open(fileName, mode='w') as csv_file:
        fieldnames = ["feature","importance"]
        #print(fieldnames)
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(cols)):
            writer.writerow({"feature":ranking[i]["feature"], "importance":ranking[i]["importance"]})
    #print(ranking)


featureRanking("FeatureRanking/giniRanking.csv", "gini")
featureRanking("FeatureRanking/entropyRanking.csv", "entropy")

