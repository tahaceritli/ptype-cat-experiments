from clevercsv import csv2df
from inputs import annotations
from src.utils_data import get_new_annotations
from src import constants

import json
import numpy as np
import pandas as pd


def read_dataset(dataset_name, data_folder):
    # because they couldn't be parsed otherwise
    DATASETS_READ_BY_CLEVERCSV = ["usp05", "Midwest_Survey_nominal", "kr-vs-kp", "Satellite", "cylinder-bands",
                                  "jungle_chess_2pcs_endgame_lion_elephant", "rodents", "wholesale-customers"]

    filename = data_folder + dataset_name + ".csv"
    if dataset_name in datasets:
        encoding, header = datasets[dataset_name]
        if dataset_name in DATASETS_READ_BY_CLEVERCSV:
            df = csv2df(filename, index_col=None, dtype=str, keep_default_na=False, skipinitialspace=True)
            df.columns = [col.replace("\"", "") for col in df.columns]
            return df
        else:
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


annotations = annotations.new_annotations
type_features = json.load(open('inputs/type_features.json', encoding="utf-8-sig",))
datasets = constants.DATASETS

X = []
y = []
z = []
for dataset in datasets:
    print(dataset)
    columns = [annotation[1] for annotation in annotations if annotation[0] == dataset]
    features = type_features[dataset]

    # the data files need to be copied here
    df = read_dataset(dataset, "inputs/files/")
    for column in columns:
        for annotation in annotations:
            if annotation[0] == dataset and annotation[1] == column:

                canonical_type = get_new_annotations(dataset, column, annotations)

                entries = list(df[column].values)
                U = len(np.unique(entries))
                N = len(entries)

                X.append(features[column])
                y.append(canonical_type)
                z.append([dataset, column, U, N, list(np.unique(entries))])

X = np.array(X, dtype=float)
y = np.array(y)
z = np.array(z, dtype=object)
np.save("inputs/X.npy", X)
np.save("inputs/y.npy", y)
np.save("inputs/z.npy", z)
