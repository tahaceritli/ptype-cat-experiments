import json
import math
import numbers
import numpy as np
import os
import pandas as pd

from collections import Counter
from nltk.corpus import wordnet as wn

from src import constants
from src.constants import FEATURE
from src.utils_data import bot_read_data

def get_scores(method, j=None, scores=None):
    scores = scores[j * 4 : j * 4 + 4]
    return {
        canonical_type: round(score, 2)
        for canonical_type, score in zip(constants.CANONICAL_TYPES_SORTED, scores)
    }


def get_bot_scores(path="bot/bot_column_type_predictions_combined.json"):
    predictions = json.load(open(path, encoding="utf-8-sig"))
    z = np.load("inputs_combined/z.npy", allow_pickle="TRUE")
    X = []
    for z_i in z:
        dataset = z_i[0]
        column = z_i[1]
        y_hat = constants.MAP_OUTPUTS[predictions[dataset][column]]

        X_i = [0.0, 0.0, 0.0, 0.0]
        index = constants.CANONICAL_TYPES_SORTED.index(y_hat)
        X_i[index] = 1.0

        X.append(X_i)

    return np.array(X, dtype=float)

######### BOT FUNCTIONS #########

# Function to automatically infer data types for a specific feature that has the standard 'object' data type
# Data types that we want to infer: boolean, date, float, integer, string
# Note that every feature that is not either a boolean, a date, a float or an integer, is classified as a string
# Input: Pandas Dataframe consisting of one single feature (so n*1 in size)
# Output: Data type of the feature (in string format)


def autoInferObject(raw_data_feature):
    dataType = ""
    types = ["date", "float64", "int64", "string"]
    weights = [0, 0, 0, 0]  # Weights corresponding to the data types

    featureLength = len(raw_data_feature)

    # numberOfIndices = 100  # Number of different values to check in a feature
    # randomIndices = random.sample(
    #     range(0, featureLength), min(numberOfIndices, featureLength)
    # )  # Array of random indices

    # If the feature only contains two different unique values, then infer it as boolean
    if len(pd.unique(raw_data_feature)) == 2:
        dataType = "bool"
    else:
        # for i in randomIndices:
        for i in range(featureLength):
            try:
                if len(raw_data_feature[i]) <= 10 and (
                    (
                        (
                            raw_data_feature[i][2:3] == "-"
                            or raw_data_feature[i][2:3] == "/"
                        )
                        and (
                            raw_data_feature[i][5:6] == "-"
                            or raw_data_feature[i][5:6] == "/"
                        )
                    )
                    or (
                        (
                            raw_data_feature[i][4:5] == "-"
                            or raw_data_feature[i][4:5] == "/"
                        )
                        and (
                            raw_data_feature[i][7:8] == "-"
                            or raw_data_feature[i][7:8] == "/"
                        )
                    )
                ):
                    weights[0] += 1  # Date
                else:
                    weights[3] += 1  # String
            except (TypeError, ValueError, IndexError):
                try:
                    int(raw_data_feature[i])
                    if "." in str(raw_data_feature[i]):
                        weights[1] += 1  # Float
                    else:
                        weights[2] += 1  # Integer
                except (TypeError, ValueError, IndexError):
                    weights[3] += 1  # String

        # print ("Date: {}, Float64: {}, Int64: {}, String: {}".format(weights[0],weights[1],weights[2],weights[3])) #For debugging purposes
        dataType = types[weights.index(max(weights))]

    return dataType


# Input: Pandas Dataframe created directly from the raw data with the pd.read_csv function
# Output: List of data types, one data type for each feature


def autoDetectDataTypes(raw_data):
    result = []

    for column in raw_data:
        if raw_data.dtypes[column] == "object":
            # print ("Trying to automatically infer the data type of the",column,"feature...") #For debugging purposes
            inferredType = autoInferObject(raw_data[column])
            result.append(inferredType)
            # print ("Result:",inferredType) #For debugging purposes
        elif raw_data.dtypes[column] == "int64":
            if len(pd.unique(raw_data[column])) == 2:
                result.append("bool")
            else:
                result.append("int64")
        else:
            # The only remaining data type is 'float64', which needs no special treatment
            result.append("float64")

    return result


# Function to check if a feature contains categorical data, specifically concerning strings
# If a feature contains at most 25 unique elements, this feature is always considered as categorical
# If a feature contains between 26 and k (where k is user specified) unique values, then the function
# calculates a similarity score between all of the different values. If this score is higher than 0.70
# a feature is also considered to be categorical (since the values at least have some relationship).
# Input: Pandas Dataframe consisting of one single feature (so n*1 in size)
#        A user-determined value k (the critical value: more than k unique values cannot be categorical)
# Output: A boolean stating whether the supplied feature is categorical or not


def autoCheckCategoricalString(raw_data_feature, k=100):
    categorical = False

    allWords = pd.unique(raw_data_feature)  # All unique strings in the feature
    similarityScores = []

    if (
        len(allWords) <= 25
    ):  # If there are less than 25 unique strings, it is categorical
        categorical = True
        # print ("Less than or equal to 25 unique strings (in this case",len(allWords),")") #For debugging purposes

    elif (
        len(allWords) <= k
    ):  # Else if there are less than k unique strings, check for similarity

        for i in range(0, len(allWords) - 1):
            for j in range(i + 1, len(allWords)):
                if pd.isnull(
                    allWords[i]
                ):  # If a string has no value (NaN), turn it into some nonsense
                    allWords[i] = "abcdef"
                elif pd.isnull(allWords[j]):
                    allWords[j] = "abcdef"

                word_1 = wn.synsets(allWords[i])
                word_2 = wn.synsets(allWords[j])

                if (
                    word_1 != [] and word_2 != []
                ):  # Calculate similarity between two non-empty words
                    similarity = wn.wup_similarity(word_1[0], word_2[0])
                    # print ("Similarity between",word_1[0],"and",word_2[0],":",similarity) #For debugging purposes
                    if not pd.isnull(similarity):
                        similarityScores.append(similarity)

        # print ("Similarity Scores:\n",similarityScores) #For debugging purposes
        # print ("Mean Similarity Score:",np.mean(similarityScores)) #For debugging purposes

        if np.mean(similarityScores) > 0.50:  # 0.50 = Critical similarity value
            categorical = True

    # print ("Categorical Feature?",categorical) #For debugging purposes

    return categorical


# Function to check if a feature contains categorical data, specifically concerning integers
# If a feature contains at most 10 unique elements, this feature is always considered as categorical
# If a feature contains between 11 and k (where k is user specified) unique values, then the function
# calculates a distance score between all of the different values. If this score is smaller than 0.1
# times the mean off all the integers a feature is also considered to be categorical (since the
# values have a relatively similar distance between each other).
# Input: Pandas Dataframe consisting of one single feature (so n*1 in size)
#        A user-determined value k (the critical value: more than k unique values cannot be categorical)
# Output: A boolean stating whether the supplied feature is categorical or not


def autoCheckCategoricalInt(raw_data_feature, k=100):
    categorical = False

    allInts = pd.unique(raw_data_feature)  # All unique integers in the feature
    distanceScores = []

    if (
        len(allInts) <= 10
    ):  # If there are less tahn 10 unique integers, it is categorical
        categorical = True
        # print ("Less than or equal to 10 unique integers (in this case",len(allInts),")") #For debugging purposes

    elif (
        len(allInts) <= k
    ):  # Else if there are less than k unique integers, check for distance

        for i in range(0, len(allInts) - 1):
            for j in range(i + 1, len(allInts)):
                distance = abs(
                    allInts[i] - allInts[j]
                )  # Calculate absolute distance between two integers
                # print ("Distance between integer",allInts[i],"and",allInts[j],":",distance) #For debugging purposes
                distanceScores.append(distance)

        # print ("Distance Scores:\n",distanceScores) #For debugging purposes
        # print ("Mean Distance Score:",np.mean(distanceScores),",should be lower than:",np.mean(allInts)) #For debugging purposes

        if np.mean(distanceScores) < (np.mean(allInts)):
            categorical = True

    # print ("Categorical Feature?",categorical) #For debuggin purposes

    return categorical


# Function to automatically decide on categoricals for every single feature in a raw data set
# Input: Pandas Dataframe created directly from the raw data with the pd.read_csv function
#        Array of data types, which is the output of step 1: autoDetectDataTypes
#        [Optional] Integer specifying the critical value for which features will be checked
#           on categoricals. That is, if the unique number of elements in a feature is higher
#           than the critical value, a feature cannot be categorical. (Default value is 100)
# Output: List of data types, one data type for each feature


def autoCheckCategoricals(raw_data, predicted, k_int=100, k_str=100):
    # print ("Checking if any of the features has categorical data...") #For debugging purposes

    for j in range(0, len(predicted)):
        if predicted[j] == "int64":
            if autoCheckCategoricalInt(raw_data.iloc[:, j], k_int):
                predicted[j] = "cat_int64"
        elif predicted[j] == "string":
            if autoCheckCategoricalString(raw_data.iloc[:, j], k_str):
                predicted[j] = "cat_string"

    return predicted


def bot_infer_types(z, k_int, k_str, data_folder="inputs/files/"):
    d2p_types = constants.DATA2PANDAS_TYPES_SORTED
    datasets = list(set([z_i[0] for z_i in z]))

    # run Bot on each dataset
    predictions = {}
    for dataset in datasets:
        df = bot_read_data(data_folder, dataset + ".csv")

        # predict types
        predicted = autoDetectDataTypes(df)
        mc_types = autoCheckCategoricals(df, predicted, k_int, k_str)
        mc_types = {
            str(column): mc_types[i] for i, column in enumerate(list(df.columns))
        }

        # store types
        predictions[dataset] = mc_types

    preds = []
    scores = []
    for dataset, column, _, _, _ in z:

        y_hat = constants.BOT_TYPES_MAPPING[predictions[dataset][column]]
        preds.append(y_hat)

        score = [1.0 if t == y_hat else 0.0 for t in d2p_types]
        scores += score

    return preds, scores


def bot_autoEncodeFeature(raw_data_feature, numberOfOccurrences=np.inf):
    oheFeature = pd.get_dummies(raw_data_feature)

    if (numberOfOccurrences != np.inf):

        dropColumns = []

        for i in range(0, len(oheFeature.columns)):
            column = oheFeature.iloc[:, i].value_counts()

            if (column[1] < numberOfOccurrences):
                dropColumns.append(i)

        oheFeature.drop(oheFeature.columns[dropColumns], axis=1, inplace=True)

    return list(oheFeature.columns)


def bot_infer_categorical_values(df, hyperparams):
    cat_values = {column:{} for column in df.columns}
    for column in df.columns:
        raw_data = df[column]
        for numberOfOccurrences in hyperparams:
            cat_values[column][numberOfOccurrences] = bot_autoEncodeFeature(raw_data, numberOfOccurrences)

    return cat_values


######### CSV2ARFF FUNCTIONS #########
def csv2arff_openml_infer_types(z, k, ptype_posteriors, data_folder="inputs/files/"):
    NUM_UNIQUE_CATEGORICAL = k
    d2p_types = constants.DATA2PANDAS_TYPES_SORTED
    datasets = list(set([z_i[0] for z_i in z]))

    # run the method on each dataset
    predictions = {}
    for dataset in datasets:
        df = bot_read_data(data_folder, dataset + ".csv")

        # predict types
        mc_types = {}
        for column in df.columns:
            column_values = df[column]
            column_is_integer = (all([isinstance(el, numbers.Number) for el in column_values]) and
                                 all([math.isnan(el) or float(el).is_integer() for el in column_values]))

            if column_is_integer:
                not_nan_values = column_values[~np.isnan(column_values)]
                unique_values = np.unique(not_nan_values)
                is_categorical = (len(unique_values) <= NUM_UNIQUE_CATEGORICAL and column_is_integer)
                if is_categorical:
                    data_type_str = 'categorical'
                else:
                    data_type_str = 'integer'
            elif all([isinstance(val, str) for val in column_values]):
                if len(set(column_values)) <= NUM_UNIQUE_CATEGORICAL:
                    data_type_str = 'categorical'
                else:
                    data_type_str = 'string'
            else:
                data_type_str = 'NUMERIC'

            mc_types[column] = data_type_str

        # store types
        predictions[dataset] = mc_types

    preds = []
    scores = []
    for dataset, column, _, _, _ in z:

        # override with ptype's output
        if predictions[dataset][column] == 'NUMERIC':
            p_t = ptype_posteriors[dataset][column]
            if p_t[FEATURE.FLOAT.value] > p_t[FEATURE.INTEGER.value]:
                y_hat = 'float'
            else:
                y_hat = 'integer'
        else:
            y_hat = predictions[dataset][column]

        preds.append(y_hat)
        # score is not entirely correct, but we are not using scores anyway.
        score = [1.0 if t == y_hat else 0.0 for t in d2p_types]
        scores += score

    return preds, scores


######### WEKA FUNCTIONS #########
def csv2arff_weka_infer_types(z, ptype_posteriors):
    d2p_types = constants.DATA2PANDAS_TYPES_SORTED
    datasets = list(set([z_i[0] for z_i in z]))

    all_weka_predictions = json.load(open("inputs/weka_type_predictions.json", encoding="utf-8-sig"))
    # run the method on each dataset
    predictions = {}
    for dataset in datasets:

        # store types
        predictions[dataset] = all_weka_predictions[dataset]

    preds = []
    scores = []
    for dataset, column, _, _, _ in z:
        # override with ptype's output
        if column not in predictions[dataset]:
            print(dataset, column)

        if predictions[dataset][column] == 'numeric':
            p_t = ptype_posteriors[dataset][column]
            if p_t[FEATURE.FLOAT.value] > p_t[FEATURE.INTEGER.value]:
                y_hat = 'float'
            else:
                y_hat = 'integer'
        else:
            y_hat = predictions[dataset][column]

        preds.append(y_hat)
        # score is not entirely correct, but we are not using scores anyway.
        score = [1.0 if t == y_hat else 0.0 for t in d2p_types]
        scores += score

    return preds, scores


######### OPENML FUNCTIONS #########
def openml_infer_categorical_values(dataset_name, path):

    df = bot_read_data(path, dataset_name + ".csv")

    column_info = []
    integer_columns = []

    for column in df.columns:
        column_values = df[column]
        column_is_integer = all(
            [isinstance(el, numbers.Number) for el in column_values]
        ) and all([math.isnan(el) or float(el).is_integer() for el in column_values])
        integer_columns.append(column_is_integer)
        if column_is_integer:
            not_nan_values = column_values[~np.isnan(column_values)]
            unique_values = np.unique(not_nan_values)
            data_type_str = [str(int(val)) for val in unique_values]
        elif all([isinstance(val, str) for val in column_values]):
            data_type_str = list(set(column_values))
        else:
            data_type_str = []

        column_info.append((column, data_type_str))

    return column_info


def unique_infer_categorical_values(d, c, y, z):
    unique_cat_vals = []
    for y_i, [dataset, column, entries] in zip(y, z):
        if dataset == d and column == c and y_i == "categorical":
            unique_cat_vals = entries
    return unique_cat_vals
