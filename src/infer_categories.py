import json
import numpy as np
import pandas as pd

from collections import OrderedDict
from src.methods import openml_infer_categorical_values, bot_infer_categorical_values
from src.utils_eval_random import (
    flatten,
    is_categorical,
    jaccard_similarity,
    nested_cv_dataset,
    run_nested_cv_dataset_categorical_values,
)
from src.infer_types import load_inputs, process_features

from ptype.Ptype import Ptype
from src import constants
ptype = Ptype()
datasets = constants.DATASETS


def evaluate_encoding(annotations, predictions):
    annotations_merged = flatten(annotations)
    predictions_merged = flatten(predictions)
    evals = []
    jaccards = []
    for annotation, prediction in zip(annotations_merged, predictions_merged):
        if set(annotation) == set(prediction):
            evals.append(1)
        else:
            evals.append(0)

        jaccards.append(jaccard_similarity(annotation, prediction))

    return evals, jaccards


def ptype_load_data(dataset):
    if dataset in ["Midwest_Survey_nominal", "usp05"]:
        filename = dataset + "_.csv"
    else:
        filename = "inputs/files/" + dataset + ".csv"
    print("expected path =", filename)
    if dataset in datasets:
        encoding, header = datasets[dataset]
    else:
        encoding = "utf-8"
        header = "infer"
    df = pd.read_csv(
        filename,
        sep=",",
        dtype=str,
        skipinitialspace=True,
        encoding=encoding,
        keep_default_na=False,
        header=header,
    )
    return df

def get_ptype_categorical_values(df):
    schema = ptype.schema_fit(df)
    normal_values = {
        col_name: col.get_normal_values() for col_name, col in schema.cols.items()
    }
    return normal_values


def setup():
    from inputs import annotations

    annotations = annotations.new_annotations

    # data, labels and features
    X, y, z = load_inputs()
    datasets = set([z_i[0] for z_i in z])

    # ptype predictions
    print("running ptype")
    predictions_ptype = {}
    for dataset in datasets:
        # print("evaluating on ", dataset)
        df = ptype_load_data(dataset)
        ptype_outputs = get_ptype_categorical_values(df)
        for col in df.columns:
            predictions_ptype[(dataset, col)] = ptype_outputs[col]

    # unique predictions
    predictions_unique = {
        (z_i[0], z_i[1]): z_i[-1] for y_i, z_i in zip(y, z) if y_i == "categorical"
    }

    # openML predictions
    predictions_openml = {}
    for dataset in datasets:
        print(dataset)
        column_info = openml_infer_categorical_values(
            dataset, "inputs/"
        )
        for col, prediction in column_info:
            if is_categorical(dataset, col, annotations):
                predictions_openml[(dataset, col)] = prediction

    # Bot predictions
    predictions_bot = json.load(open('inputs/bot_cat_predictions.json', encoding="utf-8-sig", ))


    weka_outputs = json.load(open('inputs/weka_cat_predictions.json', encoding="utf-8-sig", ))
    predictions_weka = {
        (z_i[0], z_i[1]): weka_outputs[z_i[0]][z_i[1]] for y_i, z_i in zip(y, z) if y_i == "categorical"
    }

    # true values
    annotations_dict = {
        (annotation[0], annotation[1]): annotation[3]
        for annotation in annotations
        if annotation[2] == "categorical"
    }

    # methods to evaluate
    all_predictions = OrderedDict(
        {
            "OpenML": predictions_openml,
            "Weka": predictions_weka,
            "Unique": predictions_unique,
            "ptype": predictions_ptype,
            "Bot":predictions_bot
        }
    )

    # preprocessing for the features
    X_feats = process_features(X)

    # split data using nested Cross-Validation
    datasets_cv, _, y_cv, z_cv = nested_cv_dataset(X_feats, y, z, FOLD=5)

    return all_predictions, datasets_cv, annotations_dict, y_cv, z_cv


def run(all_predictions, annotations, y_cv, z_cv):

    outputs_cv = OrderedDict()
    for method, predictions in all_predictions.items():
        print("running ", method)

        # calculate outputs
        outputs = run_nested_cv_dataset_categorical_values(
            predictions, annotations, y_cv, z_cv, method=method
        )
        if method == "Bot":
            print(outputs["best_params"])

        # store outputs
        outputs_cv[method] = {
            "y": outputs["test_true"],
            "y_hat": outputs["test_preds"],
            "y_scores": outputs["test_true_scores"],
            "y_scores_hat": outputs["test_pred_scores"],
        }

    # store outputs
    outputs_cv[method] = {
        "y": outputs["test_true"],
        "y_hat": outputs["test_preds"],
        "y_scores": outputs["test_true_scores"],
        "y_scores_hat": outputs["test_pred_scores"],
    }

    return outputs_cv


def report_results(outputs_cv):
    # adapt the code to outputs_cv
    ys = outputs_cv["Unique"]["y"]
    evals, jaccards_unique = evaluate_encoding(ys, outputs_cv["Unique"]["y_hat"])
    evals_unique = [
        round(sum(evals) / len(evals), 2),
        round(sum(jaccards_unique) / len(jaccards_unique), 2),
    ]
    evals, jaccards_openml = evaluate_encoding(ys, outputs_cv["OpenML"]["y_hat"])
    evals_openml = [
        round(sum(evals) / len(evals), 2),
        round(sum(jaccards_openml) / len(jaccards_openml), 2),
    ]
    evals, jaccards_bot = evaluate_encoding(ys, outputs_cv["Bot"]["y_hat"])
    evals_bot = [
        round(sum(evals) / len(evals), 2),
        round(sum(jaccards_bot) / len(jaccards_bot), 2),
    ]
    evals, jaccards_weka = evaluate_encoding(outputs_cv["Weka"]["y"], outputs_cv["Weka"]["y_hat"])
    evals_weka = [
        round(sum(evals) / len(evals), 2),
        round(sum(jaccards_weka) / len(jaccards_weka), 2),
    ]
    evals, jaccards_ptype = evaluate_encoding(ys, outputs_cv["ptype"]["y_hat"])
    evals_ptype = [
        round(sum(evals) / len(evals), 2),
        round(sum(jaccards_ptype) / len(jaccards_ptype), 2),
    ]

    d = {"Bot":evals_bot, "OpenML": evals_openml, "Weka": evals_weka, "Unique": evals_unique, "ptype": evals_ptype}
    # d = {"Bot":evals_bot, "Unique": evals_unique}
    df = pd.DataFrame(data=d, index=["Overall Accuracy", "Average Jaccard"])
    print(df.to_latex())

    # methods_pairs = [["Unique", "ptype"],
    #                  ["OpenML", "ptype"],
    #                  ["Weka", "ptype"],
    #                  ["Unique", "OpenML"], ]
    # jcs = {"Unique":jaccards_unique, "OpenML":jaccards_openml, "ptype":jaccards_ptype, "Weka":jaccards_weka}
    # for method1, method2 in methods_pairs:
    #     result = apply_ttest(jcs[method1], jcs[method2])
    #     print(method1, method2, result)


# from scipy import stats
#
# def apply_ttest(list1, list2):
#     return stats.ttest_rel(list1, list2)


def main():
    np.random.seed(0)

    all_predictions, datasets_cv, annotations_dict, y_cv, z_cv = setup()
    print("setup is done and will run the methods")

    outputs_cv = run(all_predictions, annotations_dict, y_cv, z_cv)
    print("run is done and will evaluate the methods")

    json.dump(outputs_cv, open("outputs/results_cat.json", "w"))
    # for method in all_predictions:
    #     print(method, all_predictions[method][("Midwest_Survey_nominal", "Education")])
    report_results(outputs_cv)


if __name__ == "__main__":
    main()
