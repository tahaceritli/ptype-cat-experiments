from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

import json
import numpy as np
import pandas as pd

from src import constants
from src.methods import bot_infer_types, csv2arff_openml_infer_types, csv2arff_weka_infer_types
from src.utils_data import update_counts
from src.utils_eval_random import (
    calculate_table,
    flatten,
    nested_cv_dataset,
    run_nested_cv_dataset,
)
from src.utils_viz import (
    plot_all_pr_curves,
    plot_confusion_matrix,
    plot_hinton,
)
from src.constants import FEATURE


def load_inputs(input_folder="inputs/"):
    X = np.load(input_folder + "X.npy")
    y = np.load(input_folder + "y.npy")
    z = np.load(input_folder + "z.npy", allow_pickle=True)
    return X, y, z


def process_features(X):
    feats = [
        FEATURE.DATE,
        FEATURE.FLOAT,
        FEATURE.INTEGER,
        FEATURE.STRING,
        FEATURE.U_RATIO,
        FEATURE.U_RATIO_CLEAN,
        FEATURE.U,
        FEATURE.U_CLEAN,
    ]
    feat_indices = [feat.value for feat in feats]
    X_feats = X[:, feat_indices]

    scaler = RobustScaler()
    # magic numbers!
    X_feats[:, -2:] = scaler.fit_transform(X_feats[:, -2:])
    return X_feats


def report_counts(datasets_cv, y_cv):
    counts = []
    for i in range(constants.FOLD):
        for j in range(constants.FOLD):
            counts = update_counts(counts, y_cv, datasets_cv, "training", i, j)
            counts = update_counts(counts, y_cv, datasets_cv, "validation", i, j)

        counts = update_counts(counts, y_cv, datasets_cv, "training", i)
        counts = update_counts(counts, y_cv, datasets_cv, "test", i)

    df = pd.DataFrame.from_records(
        counts,
        columns=["Phase", "Outer Fold", "Inner Fold", "# Datasets"]
        + constants.DATA2PANDAS_TYPES_SORTED
        + ["Total",],
    )

    return df


def vectorize_label(label):
    return [1.0 if t == label else 0.0 for t in constants.DATA2PANDAS_TYPES_SORTED]


def vectorize_labels(labels):
    return flatten([vectorize_label(label) for label in labels])


def setup():
    # data, labels and features
    X, y, z = load_inputs()

    # methods to evaluate
    methods = OrderedDict(
        {
            "Bot": {
                "clf": bot_infer_types,
                "param_grid": {
                    "k_int": [10],
                    "k_str": [25]
                },
            },
            "csv2arff (OpenML)": {
                "clf": csv2arff_openml_infer_types,
                "param_grid": {
                    "k": [10],
                },
            },
            "csv2arff (Weka)": {
                "clf": csv2arff_weka_infer_types,
                "param_grid":[]
            },
            "ptype-cat": {
                "clf": LogisticRegression(multi_class="multinomial", max_iter=10000),
                "param_grid": {"penalty": ["l2"], "C": np.logspace(-4, 4, 20)},
            },
        }
    )

    # preprocessing for the features
    X_feats = process_features(X)

    # split data using nested Cross-Validation
    datasets_cv, X_cv, y_cv, z_cv = nested_cv_dataset(X_feats, y, z, FOLD=5)

    return methods, datasets_cv, X_cv, y_cv, z_cv


def run(methods, X_cv, y_cv, z_cv):
    annotations = OrderedDict(
        {
            "y": [y_cv[str(i) + "test"] for i in range(constants.FOLD)],
            "y_scores": [
                vectorize_labels(y_cv[str(i) + "test"]) for i in range(constants.FOLD)
            ],
        }
    )

    predictions = OrderedDict()
    for method, model_specs in methods.items():
        print("running ", method)

        # calculate outputs
        outputs = run_nested_cv_dataset(model_specs, X_cv, y_cv, z_cv, method=method)

        # store outputs
        predictions[method] = {
            "best_params": outputs["best_params"],
            "y_hat": outputs["test_preds"],
            "y_scores_hat": outputs["test_pred_scores"],
        }

    return annotations, predictions


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def report_results(annotations, predictions, datasets, X, y, z):
    types = constants.DATA2PANDAS_TYPES_SORTED
    methods = list(predictions.keys())

    # # counts per fold
    report_counts(datasets, y).to_latex("outputs/tables/counts.tex", index=None)

    # classification results
    dfs = [
        calculate_table(annotations["y"], predictions[method]["y_hat"], method=method)
        for method in methods
    ]
    pd.concat(dfs, axis=1).to_latex("outputs/tables/evals.tex", index=None)

    # PR curves
    plot_all_pr_curves(
        [annotations["y_scores"] for _ in methods],
        [predictions[method]["y_scores_hat"] for method in methods],
        methods=methods,
        filepath="outputs/figures/pr_curves.png",
    )

    # confusion matrices
    for method in methods:
        plot_confusion_matrix(
            annotations["y"],
            predictions[method]["y_hat"],
            labels=types,
            filepath="outputs/figures/" + method + "_" + "confusion_matrices.png",
        )

        plot_hinton(
            annotations["y"],
            predictions[method]["y_hat"],
            labels=types,
            filepath="outputs/figures/" + method + "_" + "hinton.png",
        )

    # save for printing later
    outputs = {"X": X, "z": z, "annotations": annotations, "predictions": predictions}
    json.dump(
        outputs, open("outputs/results.json", "w"), cls=NumpyEncoder
    )


def main():
    np.random.seed(0)

    #
    methods, datasets_cv, X_cv, y_cv, z_cv = setup()

    #
    annotations, predictions = run(methods, X_cv, y_cv, z_cv)

    #
    report_results(annotations, predictions, datasets_cv, X_cv, y_cv, z_cv)


if __name__ == "__main__":
    main()
