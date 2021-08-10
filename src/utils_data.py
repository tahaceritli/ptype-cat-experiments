import ast
import numpy as np
import pandas as pd

from clevercsv import csv2df
from collections import Counter
from src import constants


def get_sequence(dataset, column, annotations):
    for item in annotations[dataset]:
        if item["header"] == column:
            return item["sequence"], item["tokens"], list(item["tags"].keys())


def get_new_annotations(dataset_q, column_q, annotations):
    for new_annotation in annotations:
        if len(new_annotation) == 3:
            dataset, column, canonical_type = new_annotation
        else:
            dataset, column, canonical_type, encodings = new_annotation
        if dataset == dataset_q and column == column_q:
            return canonical_type


def read_data(_data_path, dataset_name, dtype=str):
    if dataset_name in ["mass_6.csv", "geoplaces2.csv", "rodents.csv"]:
        encoding = "ISO-8859-1"
    else:
        encoding = "utf-8"
    
    return pd.read_csv(
        _data_path + dataset_name,
        encoding=encoding,
        dtype=dtype,
        skipinitialspace=True,
        index_col=None,
        keep_default_na=False,
        header="infer",
    )

def read_dataset(dataset_name, data_folder):
    filename = data_folder + dataset_name + ".csv"
    if dataset_name in datasets:
        encoding, header = datasets[dataset_name]
        return pd.read_csv(
            filename,
            sep=",",
            dtype=str,
            encoding=encoding,
            keep_default_na=False,
            skipinitialspace=True,
            header=header,
        )
    else:
        raise Exception(f"{filename} not known.")

def bot_read_data(_data_path, dataset_name):
    if dataset_name in ["mass_6.csv", "geoplaces2.csv", "rodents.csv"]:
        encoding = "ISO-8859-1"
    elif dataset_name in ["usp05.csv", "Midwest_Survey_nominal.csv", "kr-vs-kp.csv", "Satellite.csv", "cylinder-bands.csv",
                            "jungle_chess_2pcs_endgame_lion_elephant.csv", "rodents.csv", "wholesale-customers.csv"]:
        df = csv2df(_data_path + dataset_name, skipinitialspace=True, index_col=None)
        df.columns = [col.replace("\"", "") for col in df.columns]
        return df
    else:
        encoding = "utf-8"

    return pd.read_csv(
        _data_path + dataset_name,
        encoding=encoding,
        skipinitialspace=True,
        index_col=None,
        header="infer",
    )


def load_data(path):
    df = pd.read_csv(path)

    sentences = df["tokens"].to_list()
    sentences = [ast.literal_eval(sentence) for sentence in sentences]

    labels = df["labels"].to_list()
    labels = [ast.literal_eval(label) for label in labels]
    labels = [label[0] for label in labels]
    # labels = [[CLASSES[label[0]],] for label in labels]

    tags = df["tags"].to_list()
    tags = [ast.literal_eval(tag) for tag in tags]
    tags = [tag for tag in tags]

    return sentences, labels, tags


def save_metadata_inputs(file_name, labels, tags, tokens):
    df = pd.DataFrame({"tokens": tokens, "tags": tags, "labels": labels})
    df.to_csv(constants.DATA_ROOT + file_name + ".csv", index=False)


def get_inputs(datasets, X, y, z):
    X_new = []
    y_new = []
    z_new = []
    for dataset in datasets:
        for X_i, y_i, z_i in zip(X, y, z):
            if z_i[0] == dataset:
                X_new.append(X_i)
                y_new.append(y_i)
                z_new.append(z_i)
    X_new = np.array(X_new)
    return X_new, y_new, z_new


def prepare_inputs(datasets, x, y, index2data, filename):
    add_infos, classes, data, labels, tags, tokens = get_inputs(
        datasets, x, y, index2data
    )
    save_metadata_inputs(filename, labels, tags, tokens)

    return data, classes, add_infos


def sort_array(X, D=5):
    p_max_indices = np.argmax(X[:, :D], axis=1)
    X_sorted = ()
    for i in range(D):
        temp_indices = np.where(p_max_indices == i)[0]
        X_sorted += (X[temp_indices, :],)

    X_sorted = np.concatenate(X_sorted, axis=0)
    return X_sorted


def sort_array_t(X, y, t, D=5):
    t_indices = np.where(y == t)[0]
    X_t = X[t_indices, :].T
    X_t_feat = X[t_indices, :D].T

    p_max_indices = np.argmax(X_t_feat, axis=0)

    X_t_sorted = ()
    for i in range(D):
        temp_indices = np.where(p_max_indices == i)[0]
        X_t_sorted += (X_t[:, temp_indices],)

    X_t_sorted = np.concatenate(X_t_sorted, axis=1)

    return X_t_sorted


def sort_array_metadata(X, D=4):
    p_max_indices = np.argmax(X[:, -D:], axis=1)
    X_sorted = ()
    for i in range(D):
        temp_indices = np.where(p_max_indices == i)[0]
        X_sorted += (X[temp_indices, :],)

    X_sorted = np.concatenate(X_sorted, axis=0)
    return X_sorted


def update_data(
    data, predict_scores, metadata_model, i, phase, j=None, overwrite=False, lr=False
):
    if j is None:
        index_metadata = (metadata_model, i, phase)
        index = (i, phase)
    else:
        index_metadata = (metadata_model, i, j, phase)
        index = (i, j, phase)

    if overwrite:
        data[index] = data[index][:, -4:]
    elif lr:
        y_prob = predict_scores[index]
        n = int(len(y_prob) / 4)
        y_prob = np.array(y_prob).reshape(n, 4)

        data[index] = np.hstack((data[index], y_prob))
    else:
        if metadata_model == "roberta":
            # print(predict_scores.keys())
            # print(metadata_model, i, phase, j)
            # if j is not None:
            #     if phase == "training":
            #         phase_ = "validation"
            #     elif phase == "validation":
            #         phase_ = "training"
            #     index_metadata = ("roberta", i, j, phase_)
            y_prob = predict_scores[index_metadata]
        else:
            y_prob = predict_scores[index_metadata]
            n = int(len(y_prob) / 4)
            y_prob = np.array(y_prob).reshape(n, 4)

        data[index] = np.hstack((data[index], y_prob))

    return data


def get_counts(y, labels=constants.DATA2PANDAS_TYPES_SORTED):
    counter = Counter(y)
    return [counter[t] for t in labels] + [sum(list(counter.values()))]


def update_counts(counts, y, datasets, phase, i, j=None):
    updated_counts = counts.copy()
    if j is None:
        index = str(i) + phase
        j_value = "-"
    else:
        index = str(i) + str(j) + phase
        j_value = j + 1

    count = [phase.capitalize(), i + 1, j_value, len(datasets[index])] + get_counts(
        y[index]
    )
    updated_counts.append(count)

    return updated_counts
