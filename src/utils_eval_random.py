from src import constants
from src.utils_data import get_inputs

import joblib
import json
import numpy as np
import pandas as pd

from collections import OrderedDict
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


def nested_cv_dataset(x, y, z, FOLD=5):
    datasets = list(np.unique([z_i[0] for z_i in z]))

    kf = KFold(n_splits=FOLD)
    datasets_cv = {}
    x_cv = {}
    y_cv = {}
    z_cv = {}
    for i, (train_index, test_index) in enumerate(kf.split(datasets)):
        sets_i_training = list(np.array(datasets)[train_index])
        index = str(i) + "training"
        datasets_cv[index] = sets_i_training
        x_cv[index], y_cv[index], z_cv[index] = get_inputs(sets_i_training, x, y, z)

        sets_i_test = list(np.array(datasets)[test_index])
        index = str(i) + "test"
        datasets_cv[index] = sets_i_test
        x_cv[index], y_cv[index], z_cv[index] = get_inputs(sets_i_test, x, y, z)

        for j, (train_index, validation_index) in enumerate(kf.split(sets_i_training)):
            sets_ij_training = list(np.array(sets_i_training)[train_index])
            index = str(i) + str(j) + "training"
            datasets_cv[index] = sets_ij_training
            x_cv[index], y_cv[index], z_cv[index] = get_inputs(
                sets_ij_training, x, y, z
            )

            sets_ij_validation = list(np.array(sets_i_training)[validation_index])
            index = str(i) + str(j) + "validation"
            datasets_cv[index] = sets_ij_validation
            x_cv[index], y_cv[index], z_cv[index] = get_inputs(
                sets_ij_validation, x, y, z
            )

    return datasets_cv, x_cv, y_cv, z_cv


def filter_data(X, y):
    # magic lists of types!
    indices = [
        i for i, y_i in enumerate(y) if y_i in ["categorical", "integer", "string"]
    ]
    y_indices_binary = [
        "not-categorical" if y[i] in ["integer", "string"] else y[i] for i in indices
    ]
    return indices, X[indices, :], y_indices_binary


def train_clf(clf, X, y, index):
    clf_ = clone(clf)
    # filter cat, integer and string
    _, X_, y_ = filter_data(X[index], y[index])
    # are these allowed types of input variables? can this cause floating point differences?
    return clf_.fit(X_, y_)


def test_clf(clf, X, y, index):
    ptype_types = constants.PTYPE_TYPES_SORTED
    d2p_types = constants.DATA2PANDAS_TYPES_SORTED
    potential_cat_types = ["integer", "string"]
    X_ = X[index]
    y_ = y[index]

    # combine ptype and clf
    clf_y_probs = clf.predict_proba(X_)
    y_hat = []
    y_hat_scores = []
    for i in range(len(y_)):

        # magic numbers!
        label = ptype_types[np.argmax(X_[i, :4])]

        probs = {
            t: 0.0 if t == "categorical" else X_[i, ptype_types.index(t)]
            for t in d2p_types
        }

        if label in potential_cat_types:
            prob = clf_y_probs[i]

            # distribute probs using LR
            probs_lr = sum([probs[t] for t in potential_cat_types])
            for t in ["categorical"] + potential_cat_types:
                if t == "categorical":
                    probs[t] = probs_lr * prob[np.where(clf.classes_ == t)][0]
                else:
                    probs[t] = (
                        probs_lr
                        * prob[np.where(clf.classes_ == "not-categorical")][0]
                        * probs[t]
                    )
            # find most likely type
            label = max(probs, key=probs.get)

        y_hat.append(label)
        probs = [round(probs[t], 2) for t in d2p_types]
        y_hat_scores += probs.copy()

    return y_hat, y_hat_scores


def is_equal(x, y):
    equal = True
    for k in x.__dict__:
        try:
            res = (x.__dict__[k] == y.__dict__[k]).all()
        except AttributeError:
            res = x.__dict__[k] == y.__dict__[k]
        if res == False:
            equal = False

    return equal


def run_nested_cv_dataset(model_specs, X_cv, y_cv, z_cv, FOLD=5, method="DT"):
    clf = model_specs["clf"]
    param_grid = model_specs["param_grid"]
    outputs = {
        "training_preds": [],
        "training_pred_scores": [],
        "training_true_scores": [],
        "test_preds": [],
        "test_pred_scores": [],
        "test_true_scores": [],
        "best_params": [],
    }
    ptype_posteriors = json.load(open('inputs/type_features.json', encoding="utf-8-sig", ))
    # outer loop
    for i in range(FOLD):
        results = {}
        if method == "Bot":
            for k_int in param_grid["k_int"]:
                for k_str in param_grid["k_str"]:
                    results[(k_int, k_str)] = []

                    # inner loop
                    for j in range(FOLD):
                        index = str(i) + str(j) + "validation"
                        y_ij_validation_hat, _ = clf(z_cv[index], k_int, k_str)
                        res = accuracy_score(y_cv[index], y_ij_validation_hat)
                        results[(k_int, k_str)].append(res)

                    results[(k_int, k_str)] = np.mean(results[(k_int, k_str)])
            best_k_int, best_k_str = max(results, key=results.get)
            outputs["best_params"].append([best_k_int, best_k_str])

            training_preds_i, training_pred_scores_i = clf(
                z_cv[str(i) + "training"], k_int, k_str
            )
            outputs["training_preds"].append(training_preds_i)
            outputs["training_pred_scores"].append(training_pred_scores_i)

            preds_i, pred_scores_i = clf(z_cv[str(i) + "test"], k_int, k_str)
            outputs["test_preds"].append(preds_i)
            outputs["test_pred_scores"].append(pred_scores_i)
        elif method == "csv2arff (OpenML)":
            for k in param_grid["k"]:
                results[k] = []

                # inner loop
                for j in range(FOLD):
                    index = str(i) + str(j) + "validation"
                    y_ij_validation_hat, _ = clf(z_cv[index], k, ptype_posteriors)
                    res = accuracy_score(y_cv[index], y_ij_validation_hat)
                    results[k].append(res)

                results[k] = np.mean(results[k])
            best_k = max(results, key=results.get)
            outputs["best_params"].append([best_k])

            training_preds_i, training_pred_scores_i = clf(
                z_cv[str(i) + "training"], k, ptype_posteriors
            )
            outputs["training_preds"].append(training_preds_i)
            outputs["training_pred_scores"].append(training_pred_scores_i)

            preds_i, pred_scores_i = clf(z_cv[str(i) + "test"], k, ptype_posteriors)
            outputs["test_preds"].append(preds_i)
            outputs["test_pred_scores"].append(pred_scores_i)
        elif method == "csv2arff (Weka)":
            training_preds_i, training_pred_scores_i = clf(
                z_cv[str(i) + "training"], ptype_posteriors
            )
            outputs["training_preds"].append(training_preds_i)
            outputs["training_pred_scores"].append(training_pred_scores_i)

            preds_i, pred_scores_i = clf(z_cv[str(i) + "test"], ptype_posteriors)
            outputs["test_preds"].append(preds_i)
            outputs["test_pred_scores"].append(pred_scores_i)
        else:
            hyperparams = list(param_grid.keys())

            # for each hyperparameter setting
            for first_param in param_grid[hyperparams[0]]:
                for second_param in param_grid[hyperparams[1]]:
                    results[(first_param, second_param)] = []

                    # inner loop
                    for j in range(FOLD):
                        setattr(clf, hyperparams[0], first_param)
                        setattr(clf, hyperparams[1], second_param)

                        clf_ij = train_clf(
                            clf, X_cv, y_cv, str(i) + str(j) + "training"
                        )

                        index = str(i) + str(j) + "validation"
                        y_ij_validation_hat, _ = test_clf(clf_ij, X_cv, y_cv, index)
                        res = accuracy_score(y_cv[index], y_ij_validation_hat)
                        results[(first_param, second_param)].append(res)
                    results[(first_param, second_param)] = (
                        sum(results[(first_param, second_param)]) / 5.0
                    )
            best_param = max(results, key=results.get)

            # re-train another model with the best params
            first_param_value, second_param_value = best_param
            setattr(clf, hyperparams[0], first_param_value)
            setattr(clf, hyperparams[1], second_param_value)

            outputs["best_params"].append([first_param_value, second_param_value])

            best_clf_i = train_clf(clf, X_cv, y_cv, str(i) + "training")
            try:
                previous_best_clf_i = joblib.load("outputs/models/" + method + str(i) + ".sav")
            except FileNotFoundError:
                previous_best_clf_i = {}

            if previous_best_clf_i != {} and is_equal(previous_best_clf_i, best_clf_i):
                pass
            elif previous_best_clf_i == {}:
                joblib.dump(best_clf_i, "outputs/models/" + method + str(i) + ".sav")
            else:
                joblib.dump(best_clf_i, "outputs/models/" + method + str(i) + ".sav.new")

            preds_i, pred_scores_i = test_clf(best_clf_i, X_cv, y_cv, str(i) + "test")
            outputs["test_preds"].append(preds_i)
            outputs["test_pred_scores"].append(pred_scores_i)

            training_preds_i, training_pred_scores_i = test_clf(
                best_clf_i, X_cv, y_cv, str(i) + "training"
            )
            outputs["training_preds"].append(training_preds_i)
            outputs["training_pred_scores"].append(training_pred_scores_i)

        training_true_score = []
        for label in y_cv[str(i) + "training"]:
            temp = [0.0 for t in constants.DATA2PANDAS_TYPES_SORTED]
            temp[constants.DATA2PANDAS_TYPES_SORTED.index(label)] = 1.0
            training_true_score += temp.copy()
        outputs["training_true_scores"].append(training_true_score)

    return outputs


def get_cat_val_predictions(predictions, annotations, y_cv_i, z_cv_i, method):
    test_preds_i = []
    test_preds_i_hat = []
    test_pred_scores_i = []
    test_pred_scores_i_hat = []

    if method in ["Bot", "OpenML", "Weka", "Unique", "ptype"]:
        for y_i, z_i in zip(y_cv_i, z_cv_i):
            if y_i == "categorical":
                dataset = z_i[0]
                col = z_i[1]
                all_vals = z_i[-1]
                cat_vals = annotations[(dataset, col)]
                if method == "Weka":
                    cat_vals = [v.replace(",", "").replace('"', '').replace('\'', '') for v in cat_vals]
                    all_vals = [v.replace(",", "").replace('"', '').replace('\'', '') for v in all_vals]
                cat_val_probs = [1 if val in cat_vals else 0 for val in all_vals]

                if method == "ptype":
                    cat_vals_hat = list(predictions[(dataset, col)].keys())
                    cat_val_probs_hat = [
                        predictions[(dataset, col)][val] if val in cat_vals_hat else 0.0
                        for val in all_vals
                    ]
                else:
                    cat_vals_hat = predictions[(dataset, col)]
                    cat_val_probs_hat = [
                        1.0 if val in cat_vals_hat else 0.0 for val in all_vals
                    ]

                test_preds_i.append(cat_vals)
                test_preds_i_hat.append(cat_vals_hat)
                test_pred_scores_i = test_pred_scores_i + cat_val_probs
                test_pred_scores_i_hat = test_pred_scores_i_hat + cat_val_probs_hat
    else:
        print("Unknown method!")

    return (
        test_preds_i,
        test_preds_i_hat,
        test_pred_scores_i,
        test_pred_scores_i_hat,
    )


def get_cat_val_predictions_inner(predictions, annotations, y_cv_i, z_cv_i):
    test_preds_i = []
    test_preds_i_hat = []

    for y_i, z_i in zip(y_cv_i, z_cv_i):
        if y_i == "categorical":
            dataset = z_i[0]
            col = z_i[1]
            cat_vals = annotations[(dataset, col)]

            cat_vals_hat = predictions[(dataset, col)]

            test_preds_i.append(cat_vals)
            test_preds_i_hat.append(cat_vals_hat)

    return (
        test_preds_i,
        test_preds_i_hat,
    )


def accuracy_score_inner(ys, ys_hats):
    evals, jaccards_bot = evaluate_encoding(ys, ys_hats)
    return sum(evals) / len(evals)


def bot_infer_cat_values(predictions, z, numberOfOccurrences, data_folder="../files/"):
    datasets = list(set([z_i[0] for z_i in z]))

    # run Bot on each dataset
    all_predictions = {}
    for dataset in datasets:
        # store types
        for column in predictions[dataset]:
            # print(dataset)
            # print(column)
            # print(numberOfOccurrences)
            # print(predictions[dataset][column].keys())
            all_predictions[(dataset, column)] = predictions[dataset][column][numberOfOccurrences]

    return all_predictions

def run_nested_cv_dataset_categorical_values(
    predictions, annotations, y_cv, z_cv, FOLD=5, method="DT"
):
    outputs = {
        "test_true": [],
        "test_preds": [],
        "test_true_scores": [],
        "test_pred_scores": [],
        "best_params":{"Bot":[]}
    }
    # outer loop
    for i in range(FOLD):
        # print("outer fold ", i)
        if method == "Bot":
            results = {}
            for numberOfOccurrences in ["5", "10", "20", "40", "80"]:
                # print("numberOfOccurrences ", numberOfOccurrences)
                results[numberOfOccurrences] = []

                # inner loop
                for j in range(FOLD):
                    # print("inner fold ", j)
                    index = str(i) + str(j) + "validation"
                    predictions_index = bot_infer_cat_values(predictions, z_cv[index], numberOfOccurrences)

                    # update evaluation part and make it similar to get_cat_val_predictions
                    test_preds_i, test_preds_i_hat = get_cat_val_predictions_inner(predictions_index, annotations, y_cv[index], z_cv[index])
                    res = accuracy_score_inner(test_preds_i, test_preds_i_hat)
                    results[numberOfOccurrences].append(res)

                results[numberOfOccurrences] = np.mean(results[numberOfOccurrences])

                best_numberOfOccurrences = max(results, key=results.get)
                outputs["best_params"][method].append([best_numberOfOccurrences])

            index = str(i) + "test"
            predictions_index = bot_infer_cat_values(predictions, z_cv[index], best_numberOfOccurrences)
        else:
            predictions_index = predictions
        if method in ["Bot", "OpenML", "Weka", "Unique", "ptype"]:
            (
                test_preds_i,
                test_preds_i_hat,
                test_pred_scores_i,
                test_pred_scores_i_hat,
            ) = get_cat_val_predictions(
                predictions_index,
                annotations,
                y_cv[str(i) + "test"],
                z_cv[str(i) + "test"],
                method,
            )
            outputs["test_true"].append(test_preds_i)
            outputs["test_preds"].append(test_preds_i_hat)
            outputs["test_true_scores"].append(test_pred_scores_i)
            outputs["test_pred_scores"].append(test_pred_scores_i_hat)
        else:
            print("Unknown method!")

    return outputs


def nested_cv(X, y, FOLD=5):
    kf = StratifiedKFold(n_splits=FOLD, random_state=0)

    X_trains_outer, y_trains_outer = [], []
    X_tests, y_tests = [], []

    X_trains_inner, y_trains_inner = [], []
    X_valids, y_valids = [], []

    index2data_tests = []
    # outer loop
    for train_index, test_index in kf.split(X, y):
        X_tr, X_ts = X[train_index], X[test_index]
        y_tr, y_ts = y[train_index], y[test_index]
        index2data_tests.append(test_index)

        X_trains_outer.append(X_tr)
        y_trains_outer.append(y_tr)

        X_tests.append(X_ts)
        y_tests.append(y_ts)

        # inner loop
        X_trs, X_vals = [], []
        y_trs, y_vals = [], []
        for train_index, valid_index in kf.split(X_tr, y_tr):
            X_trs.append(X_tr[train_index])
            X_vals.append(X_tr[valid_index])
            y_trs.append(y_tr[train_index])
            y_vals.append(y_tr[valid_index])

        X_trains_inner.append(X_trs)
        y_trains_inner.append(y_trs)

        X_valids.append(X_vals)
        y_valids.append(y_vals)
    return [
        X_trains_outer,
        X_trains_inner,
        X_tests,
        X_valids,
        y_trains_outer,
        y_trains_inner,
        y_valids,
        y_tests,
        index2data_tests,
    ]


def run_nested_cv(model_specs, data_splits, FOLD=5, method="DT"):
    [
        X_trains_outer,
        X_trains_inner,
        X_tests,
        X_valids,
        y_trains_outer,
        y_trains_inner,
        y_valids,
        y_tests,
        index2data_tests,
    ] = data_splits
    model = model_specs["clf"]
    param_grid = model_specs["param_grid"]
    preds = []
    pred_scores = []
    true_scores = []
    for i in range(FOLD):
        # param search
        results = {}
        if method == "ptype":
            preds_i, pred_scores_i = model(X_tests[i])
            preds.append(preds_i)
            pred_scores.append(pred_scores_i)
        else:
            hyperparams = list(param_grid.keys())

            for first_param in param_grid[hyperparams[0]]:
                for second_param in param_grid[hyperparams[1]]:

                    results[(first_param, second_param)] = []
                    for X_tr, X_val, y_tr, y_val in zip(
                        X_trains_inner[i], X_valids[i], y_trains_inner[i], y_valids[i],
                    ):
                        setattr(model, hyperparams[0], first_param)
                        setattr(model, hyperparams[1], second_param)

                        res = model.fit(X_tr, y_tr).score(X_val, y_val)
                        results[(first_param, second_param)].append(res)
                    results[(first_param, second_param)] = (
                        sum(results[(first_param, second_param)]) / 5.0
                    )
            best_param = max(results, key=results.get)

            # re-train another model with the best params
            first_param_value, second_param_value = best_param
            setattr(model, hyperparams[0], first_param_value)
            setattr(model, hyperparams[1], second_param_value)

            clf = model.fit(X_trains_outer[i], y_trains_outer[i])
            filename = "models/" + method + str(i) + ".sav"
            joblib.dump(clf, filename)

            preds.append(clf.predict(X_tests[i]))
            pred_scores.append(list(clf.predict_proba(X_tests[i]).flatten()))

        true_score = []
        for label in y_tests[i]:
            temp = [0.0, 0.0, 0.0, 0.0]
            temp[constants.DATA2PANDAS_TYPES_SORTED.index(label)] = 1.0
            true_score += temp.copy()
        true_scores.append(true_score)

    return pred_scores, true_scores, preds, y_tests


def flatten(l):
    return [item for sublist in l for item in sublist]


def calculate_table(y_tests, y_tests_hat, method, separately=False):
    true_labels = y_tests.copy()
    predicted_labels = y_tests_hat.copy()

    if not separately:
        true_labels = flatten(y_tests)
        predicted_labels = flatten(y_tests_hat)

    df, J, overall_accuracy = evaluate_method(
        _annotations=true_labels, _predictions=predicted_labels
    )

    results_table = OrderedDict()
    for d2p_type in constants.DATA2PANDAS_TYPES_SORTED:
        results_table[d2p_type] = float(J[d2p_type])
    results_table["overall accuracy"] = overall_accuracy

    # export results
    df = pd.DataFrame.from_dict(results_table, orient="index")
    df = df.reindex(["overall accuracy"] + constants.DATA2PANDAS_TYPES_SORTED).round(3)
    df.columns = [method]
    return df


def calculate_table_macro(y_tests, y_tests_hat, method):
    results_table = OrderedDict()
    for can_type in constants.DATA2PANDAS_TYPES_SORTED:
        results_table[can_type] = []
    results_table["overall accuracy"] = []

    for y_test, y_test_hat in zip(y_tests, y_tests_hat):
        df, J, overall_accuracy = evaluate_method(
            _annotations=y_test, _predictions=y_test_hat
        )
        for can_type in constants.DATA2PANDAS_TYPES_SORTED:
            results_table[can_type].append(float(J[can_type]))
        results_table["overall accuracy"].append(overall_accuracy)
    for can_type in constants.DATA2PANDAS_TYPES_SORTED:
        results_table[can_type] = sum(results_table[can_type]) / 5.0
    results_table["overall accuracy"] = sum(results_table["overall accuracy"]) / 5.0

    # export results
    df = pd.DataFrame.from_dict(results_table, orient="index")
    df = df.reindex(["overall accuracy"] + constants.DATA2PANDAS_TYPES_SORTED).round(2)
    df.columns = [method]
    return df


def run_smoothed(X_trainings, y_trainings, X_tests, alpha=1, C=4):
    y_tests_hat_smoothed = []
    pred_scores_hat_smoothed = []
    for i, (X_training, y_training, X_test) in enumerate(
        zip(X_trainings, y_trainings, X_tests)
    ):
        pred = []
        pred_scores = []

        clf = joblib.load("models/DecisionTree" + str(i) + ".sav")

        leave_ids_training = clf.apply(X_training)
        leave_ids_test = clf.apply(X_test)

        node_2_ids = {}
        for i, node_id in enumerate(leave_ids_training):
            if node_id not in node_2_ids:
                node_2_ids[node_id] = [y_training[i]]
            else:
                node_2_ids[node_id].append(y_training[i])

        for node_id in leave_ids_test:
            node_elements = [] if node_id not in node_2_ids else node_2_ids[node_id]

            prob = []
            for t in clf.classes_:
                x_i = node_elements.count(t)
                prob.append((x_i + alpha) / (x_i + C * alpha))
            prob = np.array(prob) / sum(prob)

            pred.append(constants.DATA2PANDAS_TYPES_SORTED[np.argmax(prob)])
            pred_scores += list(prob)

        y_tests_hat_smoothed.append(pred)
        pred_scores_hat_smoothed.append(pred_scores)
    return y_tests_hat_smoothed, pred_scores_hat_smoothed


def post_prune(X_trains, y_trains, X_tests, y_tests):
    y_tests_hat_pruned = []
    for i, (X_train, y_train, X_test, y_test) in enumerate(
        zip(X_trains, y_trains, X_tests, y_tests)
    ):

        filename = "models/DT" + str(i) + ".sav"
        clf = joblib.load(filename)
        max_leaf_nodes = int(clf.max_leaf_nodes)
        max_depth = int(clf.max_depth)
        path = clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, _ = path.ccp_alphas, path.impurities

        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(
                max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, ccp_alpha=ccp_alpha,
            )
            clf.fit(X_train, y_train)
            clfs.append(clf)

        test_scores = [clf.score(X_test, y_test) for clf in clfs]
        best_ccp_alpha = ccp_alphas[np.argmax(test_scores)]

        clf = DecisionTreeClassifier(
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            ccp_alpha=best_ccp_alpha,
        )
        clf.fit(X_train, y_train)
        y_tests_hat_pruned.append(clf.predict(X_test))

    return y_tests_hat_pruned


def not_vector(X):
    return np.array([not x for x in X])


def evaluate_model_type(annotations, predictions, t):
    y_true = np.array(annotations) == t
    y_score = np.array(predictions) == t

    tp = sum(y_true * y_score)
    fp = sum(not_vector(y_true) * y_score)
    fn = sum(y_true * not_vector(y_score))

    return [tp, fp, fn]


def get_evaluations(
    _annotations,
    _predictions,
    types=["categorical", "date", "float", "integer", "string"],
):

    J = {}
    for t in types:
        [tp, fp, fn] = evaluate_model_type(_annotations, _predictions, t)
        J[t] = "{:.2f}".format(tp / (tp + fp + fn))

    overall_accuracy = [
        1 if annotation == prediction else 0
        for annotation, prediction in zip(_annotations, _predictions)
    ]
    overall_accuracy = sum(overall_accuracy) / len(overall_accuracy)

    return J, overall_accuracy


def evaluate_method(
    _annotations,
    _predictions,
    classes=["categorical", "date", "float", "integer", "string"],
):
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for annotation, prediction in zip(_annotations, _predictions):
        if prediction != "any":
            confusion_matrix[classes.index(prediction), classes.index(annotation)] += 1

    # colon'lar gercek label'lar, row'lar predicted'lar
    df = pd.DataFrame.from_records(confusion_matrix, columns=classes)
    df.index = classes

    J, overall_accuracy = get_evaluations(_annotations, _predictions)

    return df, J, overall_accuracy


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def evaluate_encoding(annotations, predictions):
    evals = []
    jaccards = []
    for annotation, prediction in zip(annotations, predictions):
        if set(annotation) == set(prediction):
            evals.append(1)
        else:
            evals.append(0)

        jaccards.append(jaccard_similarity(annotation, prediction))

    return evals, jaccards


def get_annotation(dataset, column, annotations):
    for annotation in annotations:
        if annotation[0] == dataset and annotation[1] == column:
            return annotation[-1]


def get_cat_val_annotation(dataset, column, annotations):
    for annotation in annotations:
        if (
            annotation[0] == dataset
            and annotation[1] == column
            and annotation[2] == "categorical"
        ):
            return annotation[-1]


def is_categorical(dataset, column, annotations):
    is_cat = False
    for annotation in annotations:
        if (
            annotation[0] == dataset
            and annotation[1] == column
            and annotation[2] == "categorical"
        ):
            is_cat = True
    return is_cat


def get_prediction(dataset, column, predictions):
    for prediction in predictions:
        if prediction[0] == dataset and prediction[1] == column:
            return prediction[-1]


def get_prediction_ptype(dataset, column, predictions):
    return predictions[dataset + ".csv"][column][3]


def get_prediction_prob(dataset, column, predictions):
    return predictions[dataset + ".csv"][column][2]
