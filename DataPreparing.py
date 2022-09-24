import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler
import pickle


def save_data(obj, filename):
    file = open(filename, 'wb')
    pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
    file.close()


def load_data(filename):
    file = open(filename,'rb')
    obj = pickle.load(file)
    file.close()
    X = obj["features"]
    Y = obj["labels"]
    cols = obj["cols"]
    return X, Y, cols


def preprocess(filename, object_filename):
    df = pd.read_csv(filename, delimiter=",")
    columns = []
    for i in range(8):
        columns.append(i)
    columns.append(-1)

    df.drop(df.columns[columns], axis=1, inplace=True)
    Y = copy.deepcopy(df["tp"]).values
    Y = Y.astype(int)
    df.drop(df.columns[[-1]], axis=1, inplace=True)
    df.drop(["file_size","number_of_line_with_comment","buggy_method_starting_line","buggy_method_ending_line"
            ,"buggy_method_length_with_comment","buggy_class_start_line_with_comment",'buggy_class_end_line_with_comment', 'buggy_class_length_with_comment'
            ,'buggy_method_start_line_without_comment',
            'buggy_method_end_line_without_comment',
            'buggy_class_start_line_without_comment',
            'buggy_class_end_line_without_comment',
            'buggy_method_length_without_comment',
            'buggy_class_length_without_comment',
            'buggy_file_length_without_comment'], axis=1, inplace=True)
    #df.drop(["number_of_line_with_comment"], axis=1, inplace=True)

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X = X.values
    cols = df.columns
    obj = {"features": X, "labels": Y, "cols":cols}
    save_data(obj,object_filename)

    return X, Y, cols


if __name__ == "__main__":
    filename = "Data/OWASP_Trainable_Dataset.csv"
    object_filename = "Data/OWASP_Trainable_Refined_Dataset.pkl"

    #preprocess(filename, object_filename)
    X,Y,cols = load_data(object_filename)
    
    print(X.shape[0], Y.shape)
    print(cols)
    print(len(cols))
