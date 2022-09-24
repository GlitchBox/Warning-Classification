import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler
import pickle
import csv


def save_data(obj, filename):
    file = open(filename, 'wb')
    pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
    file.close()


def load_data(filename):
    file = open(filename, 'rb')
    obj = pickle.load(file)
    file.close()
    X = obj["features"]
    Y = obj["labels"]
    cols = obj["cols"]
    return X, Y, cols


def PreprocessTP(fileName, object_fileName):
    csvFile = open(fileName, mode="r")
    csvReader = csv.DictReader(csvFile)

    dataList = []
    fieldNames = []
    line = 0
    for row in csvReader:
        newDict = {}
        if (row["tp"] == "1"):
            for key in row:
                if (line == 0 and key != "Categories"):
                    fieldNames.append(key)

                if (key == "Categories"):
                    if (row[key] == "Critical"):
                        newDict["label"] = 1
                    elif (row[key] == "Major" or row[key] == "Minor"):
                        newDict["label"] = 0
                else:
                    newDict[key] = row[key]
            dataList.append(newDict)
        line = line + 1

    csvFile.close()
    fieldNames.append("label")
    print("Reading done")

    csvFile = open(object_fileName, mode="w")
    csvWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
    csvWriter.writeheader()

    for row in dataList:
        csvWriter.writerow(row)
    csvFile.close()


def PreprocessCombined(fileName, object_fileName):
    csvFile = open(fileName, mode="r")
    csvReader = csv.DictReader(csvFile)

    dataList = []
    fieldNames = []
    line = 0
    for row in csvReader:
        newDict = {}
        for key in row:
            if (line == 0 and key != "Categories"):
                fieldNames.append(key)

            if (key == "Categories"):
                if (row[key] == "Critical"):
                    newDict["label"] = 1
                elif (row[key] == "Major" or row[key] == "Minor"):
                    newDict["label"] = 2
                else:
                    newDict["label"] = 0
            else:
                newDict[key] = row[key]
        dataList.append(newDict)
        line = line + 1

    csvFile.close()
    fieldNames.append("label")
    print("Reading done")

    csvFile = open(object_fileName, mode="w")
    csvWriter = csv.DictWriter(csvFile, fieldnames=fieldNames)
    csvWriter.writeheader()

    for row in dataList:
        csvWriter.writerow(row)
    csvFile.close()


def preprocess(filename, object_filename):
    df = pd.read_csv(filename, delimiter=",")
    columns = []
    for i in range(8):
        columns.append(i)
    # columns.append(-1)

    df.drop(df.columns[columns], axis=1, inplace=True)  # removes columns 0 through 7
    Y = copy.deepcopy(df["label"]).values
    Y = Y.astype(int)
    df.drop(df.columns[[-1]], axis=1, inplace=True)  # removes label
    df.drop(
        ["short_description", "number_of_line_with_comment", "buggy_method_starting_line", "buggy_method_ending_line"
            , "buggy_method_length_with_comment", "buggy_class_start_line_with_comment",
         'buggy_class_end_line_with_comment', 'buggy_class_length_with_comment'
            , 'buggy_method_start_line_without_comment',
         'buggy_method_end_line_without_comment',
         'buggy_class_start_line_without_comment',
         'buggy_class_end_line_without_comment',
         'buggy_method_length_without_comment',
         'buggy_class_length_without_comment',
         'buggy_file_length_without_comment', 'tp', 'fp', 'Type'], axis=1, inplace=True)

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X = X.values
    cols = df.columns
    obj = {"features": X, "labels": Y, "cols": cols}
    save_data(obj, object_filename)

    return X, Y, cols


if __name__ == "__main__":
    filename = "Data/Pure_Final_Trainable_Dataset.csv"
    object_filename = "Data/Combined_Dataset.pkl"
    object_filename1 = "Data/TP_Dataset.pkl"
    object_filename2 = "Data/OWASP_Trainable_Combined_Dataset.csv"
    object_filename3 = "Data/OWASP_Trainable_TP_Dataset.csv"

    # PreprocessCombined(fileName=filename, object_fileName= object_filename2)
    # PreprocessTP(fileName=filename, object_fileName=object_filename3)

    preprocess(object_filename3, object_filename1)
    X, Y, cols = load_data(object_filename1)
    print("TP dataset")
    print(X.shape, Y.shape)
    print(cols)
    print(len(cols))

    preprocess(object_filename2, object_filename)
    X, Y, cols = load_data(object_filename)
    print("Combined Dataset")
    print(X.shape, Y.shape)
    print(cols)
    print(len(cols))
