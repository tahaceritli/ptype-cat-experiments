from src.methods import bot_infer_categorical_values
from src.utils_data import bot_read_data

import numpy as np
import json

data_folder = "inputs/"
y = np.load(data_folder + "y.npy")
z = np.load(data_folder + "z.npy", allow_pickle=True)
datasets = list(set([z_i[0] for z_i in z]))


def filter_categorical_columns(dataset, df, y, z):
    columns = []
    for y_i, z_i in zip(y,z):
        if z_i[0] == dataset:
            if y_i == "categorical":
                columns.append(z_i[1])
    return df[columns]


hyperparams = [5, 10, 20, 40, 80]
bot_predictions = {dataset:{} for dataset in datasets}

for dataset in ["ldpa"]:
    print("reading ", dataset)
    df = bot_read_data(data_folder + "files/", dataset + ".csv")
    df = filter_categorical_columns(dataset, df, y, z)
    bot_predictions[dataset] = bot_infer_categorical_values(df, hyperparams)

# this is saved to the inputs folder because it'll be loaded after
json.dump(bot_predictions, open("inputs/bot_cat_predictions.json", "w"))

