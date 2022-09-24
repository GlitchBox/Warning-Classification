import os

import numpy as np
import pandas


def calculate_f1_score_using_panda(file_name):
    """

    :param file_name:
    """
    df = pandas.read_csv(file_name)
    # print(df)

    # print((df['precision'] + 1))
    df['f1_score'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])

    df['f1_score'].fillna(0.0, inplace=True)
    df = df.replace(to_replace=np.inf, value=0.0)
    df = df.sort_values(by='f1_score', ascending=False)

    # print(df['f1_score'])
    df.to_csv(file_name, mode='w', index=False)


if __name__ == "__main__":
    for root, dirs, files in os.walk(
            "/mnt/LocalDiskG/ML/warning classification/Warning Classification/ModelResults/LightGBMResults/LightGBMRegressor"):
        for file in files:
            if file.endswith(".csv"):
                file_name = os.path.join(root, file)
                print(file_name)
                calculate_f1_score_using_panda(file_name)
