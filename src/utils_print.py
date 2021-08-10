import json
import numpy as np
import os

from src import constants
from src.methods import get_scores


def append_textfile(filepath, txt):
    with open(filepath, "a") as myfile:
        myfile.write(txt)


def remove_existing_files(method):
    for t in constants.CANONICAL_TYPES_SORTED:
        if os.path.exists("outputs/failure_cases/" + method + "_" + t + ".txt"):
            os.remove("outputs/failure_cases/" + method + "_" + t + ".txt")
        if os.path.exists("outputs/success_cases/" + method + "_" + t + ".txt"):
            os.remove("outputs/success_cases/" + method + "_" + t + ".txt")


def get_text(
    i,
    dataset,
    column,
    entries,
    seq,
    annotation,
    keyword_prob,
    bi_lstm_prob,
    bi_lstm_attention_prob,
):
    txt = (
        "Dataset = "
        + dataset
        + "\nColumn = "
        + column
        + "\nFold = "
        + str(i)
        + "\n# Entries = "
        + str(len(entries))
        + "\n# Unique Entries = "
        + str(len(np.unique(entries)))
        + "\nSome Unique Data Entries = ["
        + ", ".join(np.unique(entries)[:5])
        + "]"
        + "\nMetadata Sequence = "
        + seq
        + "Annotation = "
        + annotation
        + "\nKeyword Search probabilities = "
        + json.dumps(keyword_prob).replace('"', "")
        + "\nBi-LSTM posterior = "
        + json.dumps(bi_lstm_prob).replace('"', "")
        + "\nBi-LSTM-Attention posterior = "
        + json.dumps(bi_lstm_attention_prob).replace('"', "")
        + "\n\n"
    )

    return txt


def append_to_file(
    prediction,
    i,
    dataset,
    column,
    entries,
    seq,
    annotation,
    keyword_prob,
    bi_lstm_prob,
    bi_lstm_attention_prob,
    method,
):
    txt = get_text(
        i,
        dataset,
        column,
        entries,
        seq,
        annotation,
        keyword_prob,
        bi_lstm_prob,
        bi_lstm_attention_prob,
    )

    if annotation != prediction:
        append_textfile("outputs/failure_cases/" + method + annotation + ".txt", txt)
    else:
        append_textfile("outputs/success_cases/" + method + annotation + ".txt", txt)


def remove_old_failure_cases(summary_folder, methods):
    for t in constants.DATA2PANDAS_TYPES_SORTED:
        for method in methods:
            if os.path.exists(summary_folder + t + "_" + method + "_wins.txt"):
                os.remove(summary_folder + t + "_" + method + "_wins.txt")
            if os.path.exists(summary_folder + method + "_" + t + ".txt"):
                os.remove(summary_folder + method + "_" + t + ".txt")


def get_y_hat_probs(predictions, i, j):
    types = constants.DATA2PANDAS_TYPES_SORTED
    C = len(types)

    y_hat_probs = {}
    for method in predictions:
        scores = predictions[method]["y_scores_hat"][i][j * C : j * C + C]
        y_hat_probs[method] = {t: round(s, 2) for t, s in zip(types, scores)}

    return y_hat_probs
    # ptype_scores = {
    #     t: round(score, 2)
    #     for t, score in zip(constants.PTYPE_TYPES_SORTED, X_tests_i[j, :])
    # }


def get_txt(i, features, y, z, y_hat, scores):
    types = constants.PTYPE_TYPES_SORTED
    ptype_probs = {t: features[t_i] for t_i, t in enumerate(types)}
    dataset, column, U, N, entries = z
    return (
        "Dataset = "
        + dataset
        + "\nColumn = "
        + column
        + "\nFold = "
        + str(i)
        + "\n# Entries = "
        + str(N)
        + "\n# Unique Entries = "
        + str(U)
        + "\nSome Unique Data Entries = ["
        + ", ".join(np.unique(entries)[:5])
        + "]"
        + "\nAnnotation = "
        + y
        + "\nptype probs = "
        + json.dumps(ptype_probs).replace('"', "")
        + "\nBot Prediction = "
        + y_hat["Bot"]
        + "\nBot Scores = "
        + json.dumps(scores["Bot"]).replace('"', "")
        + "\ndata2pandas Prediction = "
        + y_hat["data2pandas"]
        + "\ndata2pandas Scores= "
        + json.dumps(scores["data2pandas"]).replace('"', "")
        + "\n\n"
    )


def get_best_type(scores):
    return max(scores, key=scores.get)


def print_failure_cases(outputs):
    X_tests = [outputs["X"][str(i) + "test"] for i in range(constants.FOLD)]
    y_tests = outputs["annotations"]["y"]
    z_tests = [outputs["z"][str(i) + "test"] for i in range(constants.FOLD)]
    predictions = outputs["predictions"]

    methods = list(predictions.keys())
    summary_folder = "tests/outputs/failure_cases/comparisons/"
    remove_old_failure_cases(summary_folder, methods)

    for i in range(constants.FOLD):
        for j in range(len(X_tests[i])):

            features = X_tests[i][j]
            y = y_tests[i][j]

            y_hat_probs = get_y_hat_probs(predictions, i, j)
            y_hat = {
                method: get_best_type(y_hat_probs[method]) for method in y_hat_probs
            }
            z = z_tests[i][j]

            # evaluate method
            txt = get_txt(i, features, y, z, y_hat, y_hat_probs)

            if y_hat["Bot"] == y and y_hat["data2pandas"] != y:
                filepath = summary_folder + y + "_Bot_wins.txt"
                append_textfile(filepath, txt)
            elif y_hat["Bot"] != y and y_hat["data2pandas"] == y:
                filepath = summary_folder + y + "_data2pandas_wins.txt"
                append_textfile(filepath, txt)

            if y_hat["data2pandas"] != y:
                filepath = summary_folder + "data2pandas_" + y + ".txt"
                append_textfile(filepath, txt)


def print_failure_cases_dt_lr(
    X_tests, trues, all_preds, all_pred_scores, dataset_column_zs, X, z
):
    for t in constants.CANONICAL_TYPES_SORTED:
        if os.path.exists("outputs/failure_cases/DT_LR_" + t + ".txt"):
            os.remove("outputs/failure_cases/DT_LR_" + t + ".txt")
        if os.path.exists("outputs/failure_cases/LR_DT_" + t + ".txt"):
            os.remove("outputs/failure_cases/LR_DT_" + t + ".txt")

    for i in range(len(X_tests)):
        annotations = trues[i]

        predictions_ptype = all_preds[1][i]
        predictions_DT = all_preds[2][i]
        predictions_LR = all_preds[3][i]

        for (
            j,
            (annotation, prediction_ptype, prediction_LR, prediction_DT),
        ) in enumerate(
            zip(annotations, predictions_ptype, predictions_LR, predictions_DT)
        ):
            ptype_scores = all_pred_scores[1][i]
            ptype_prob = get_scores("ptype", j=j, scores=ptype_scores)

            DT_scores = all_pred_scores[2][i]
            DT_prob = get_scores("DecisionTree", j=j, scores=DT_scores)

            LR_scores = all_pred_scores[3][i]
            LR_prob = get_scores("LogisticRegression", j=j, scores=LR_scores)

            z_i = dataset_column_zs[i][j]
            dataset, column, unique_entries = z_i
            index = sum(
                [
                    index_i
                    for index_i, z_i_temp in enumerate(z)
                    if (z_i_temp == z_i).all()
                ]
            )
            features = X[index, :]
            U = int(features[7])
            N = int(features[9])
            if annotation == prediction_DT and prediction_LR != prediction_DT:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(unique_entries)[:5])
                    + "]"
                    + "\nU = "
                    + str(U)
                    + "\nN = "
                    + str(N)
                    + "\nAnnotation = "
                    + annotation
                    + "\nptype    posterior = "
                    + json.dumps(ptype_prob).replace('"', "")
                    + "\nLog Reg  posterior = "
                    + json.dumps(LR_prob).replace('"', "")
                    + "\nDec Tree posterior = "
                    + json.dumps(DT_prob).replace('"', "")
                    + "\n\n"
                )
                append_textfile(
                    "outputs/failure_cases/DT_LR_" + annotation + ".txt", txt
                )
            if annotation == prediction_LR and prediction_LR != prediction_DT:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(unique_entries)[:5])
                    + "]"
                    + "\nU = "
                    + str(U)
                    + "\nN = "
                    + str(N)
                    + "\nAnnotation = "
                    + annotation
                    + "\nptype    posterior = "
                    + json.dumps(ptype_prob).replace('"', "")
                    + "\nLog Reg  posterior = "
                    + json.dumps(LR_prob).replace('"', "")
                    + "\nDec Tree posterior = "
                    + json.dumps(DT_prob).replace('"', "")
                    + "\n\n"
                )
                append_textfile(
                    "outputs/failure_cases/LR_DT_" + annotation + ".txt", txt
                )


def print_failure_cases_bot_lr(
    X_tests, trues, all_preds, all_pred_scores, dataset_column_zs, X, z
):
    for t in constants.CANONICAL_TYPES_SORTED:
        if os.path.exists("outputs/failure_cases/bot_LR_" + t + ".txt"):
            os.remove("outputs/failure_cases/bot_LR_" + t + ".txt")
        if os.path.exists("outputs/failure_cases/LR_bot_" + t + ".txt"):
            os.remove("outputs/failure_cases/LR_bot_" + t + ".txt")

    for i in range(len(X_tests)):
        annotations = trues[i]

        predictions_bot = all_preds[0][i]
        predictions_LR = all_preds[3][i]

        for (j, (annotation, prediction_bot, prediction_LR),) in enumerate(
            zip(annotations, predictions_bot, predictions_LR)
        ):
            bot_scores = all_pred_scores[0][i]
            bot_prob = get_scores("bot", j=j, scores=bot_scores)

            LR_scores = all_pred_scores[3][i]
            LR_prob = get_scores("LogisticRegression", j=j, scores=LR_scores)

            z_i = dataset_column_zs[i][j]
            dataset, column, unique_entries = z_i
            index = sum(
                [
                    index_i
                    for index_i, z_i_temp in enumerate(z)
                    if (z_i_temp == z_i).all()
                ]
            )
            features = X[index, :]
            U = int(features[7])
            N = int(features[9])
            if annotation == prediction_bot and prediction_LR != prediction_bot:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(unique_entries)[:5])
                    + "]"
                    + "\nU = "
                    + str(U)
                    + "\nN = "
                    + str(N)
                    + "\nAnnotation = "
                    + annotation
                    + "\nbot    posterior = "
                    + json.dumps(bot_prob).replace('"', "")
                    + "\nLog Reg  posterior = "
                    + json.dumps(LR_prob).replace('"', "")
                    + "\n\n"
                )
                append_textfile(
                    "outputs/failure_cases/bot_LR_" + annotation + ".txt", txt
                )

            if annotation == prediction_LR and prediction_LR != prediction_bot:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(unique_entries)[:5])
                    + "]"
                    + "\nU = "
                    + str(U)
                    + "\nN = "
                    + str(N)
                    + "\nAnnotation = "
                    + annotation
                    + "\nbot    posterior = "
                    + json.dumps(bot_prob).replace('"', "")
                    + "\nLog Reg  posterior = "
                    + json.dumps(LR_prob).replace('"', "")
                    + "\n\n"
                )
                append_textfile(
                    "outputs/failure_cases/LR_bot_" + annotation + ".txt", txt
                )


def print_failure_cases_bot_dtandlr(
    X_tests, trues, all_preds, all_pred_scores, dataset_column_zs, X, z
):
    for t in constants.CANONICAL_TYPES_SORTED:
        if os.path.exists("outputs/failure_cases/bot_dtandlr_" + t + ".txt"):
            os.remove("outputs/failure_cases/bot_dtandlr_" + t + ".txt")
        if os.path.exists("outputs/failure_cases/dtandlr_bot_" + t + ".txt"):
            os.remove("outputs/failure_cases/dtandlr_bot_" + t + ".txt")

    for i in range(len(X_tests)):
        annotations = trues[i]

        predictions_bot = all_preds[0][i]
        predictions_dtandlr = all_preds[4][i]

        for (j, (annotation, prediction_bot, prediction_dtandlr),) in enumerate(
            zip(annotations, predictions_bot, predictions_dtandlr)
        ):
            bot_scores = all_pred_scores[0][i]
            bot_prob = get_scores("bot", j=j, scores=bot_scores)

            dtandlr_scores = all_pred_scores[4][i]
            dtandlr_prob = get_scores("dt+lr", j=j, scores=dtandlr_scores)

            z_i = dataset_column_zs[i][j]
            dataset, column, unique_entries = z_i
            index = sum(
                [
                    index_i
                    for index_i, z_i_temp in enumerate(z)
                    if (z_i_temp == z_i).all()
                ]
            )
            features = X[index, :]
            U = int(features[7])
            N = int(features[9])

            if annotation == prediction_bot and prediction_dtandlr != prediction_bot:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(unique_entries)[:5])
                    + "]"
                    + "\nU = "
                    + str(U)
                    + "\nN = "
                    + str(N)
                    + "\nAnnotation = "
                    + annotation
                    + "\nbot    posterior = "
                    + json.dumps(bot_prob).replace('"', "")
                    + "\nDT+LR posterior = "
                    + json.dumps(dtandlr_prob).replace('"', "")
                    + "\n\n"
                )
                append_textfile(
                    "outputs/failure_cases/bot_dtandlr_" + annotation + ".txt", txt
                )

            if (
                annotation == prediction_dtandlr
                and prediction_dtandlr != prediction_bot
            ):
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(unique_entries)[:5])
                    + "]"
                    + "\nU = "
                    + str(U)
                    + "\nN = "
                    + str(N)
                    + "\nAnnotation = "
                    + annotation
                    + "\nbot    posterior = "
                    + json.dumps(bot_prob).replace('"', "")
                    + "\nDT+LR posterior = "
                    + json.dumps(dtandlr_prob).replace('"', "")
                    + "\n\n"
                )
                append_textfile(
                    "outputs/failure_cases/dtandlr_bot_" + annotation + ".txt", txt
                )

    # print_failure_cases_dtandlr_ptype(
    #     X_tests, y_tests, all_y_tests_hat, all_pred_scores, dataset_column_zs_test, X, z
    # )


def print_failure_cases_dtandlr_ptype(
    X_tests, trues, all_preds, all_pred_scores, dataset_column_zs, X, z
):
    for t in constants.CANONICAL_TYPES_SORTED:
        if os.path.exists("outputs/failure_cases/dtandlr_ptype_" + t + ".txt"):
            os.remove("outputs/failure_cases/dtandlr_ptype_" + t + ".txt")
        if os.path.exists("outputs/failure_cases/ptype_dtandlr_" + t + ".txt"):
            os.remove("outputs/failure_cases/ptype_dtandlr_" + t + ".txt")

    for i in range(len(X_tests)):
        annotations = trues[i]

        predictions_ptype = all_preds[1][i]
        predictions_dtandlr = all_preds[4][i]

        for (j, (annotation, prediction_dtandlr, prediction_ptype),) in enumerate(
            zip(annotations, predictions_dtandlr, predictions_ptype)
        ):
            dtandlr_scores = all_pred_scores[4][i]
            dtandlr_prob = get_scores("dtandlr", j=j, scores=dtandlr_scores)

            ptype_scores = all_pred_scores[1][i]
            ptype_prob = get_scores("ptype", j=j, scores=ptype_scores)

            z_i = dataset_column_zs[i][j]
            dataset, column, unique_entries = z_i
            index = sum(
                [
                    index_i
                    for index_i, z_i_temp in enumerate(z)
                    if (z_i_temp == z_i).all()
                ]
            )
            features = X[index, :]
            U = int(features[7])
            N = int(features[9])

            if (
                annotation == prediction_dtandlr
                and prediction_ptype != prediction_dtandlr
            ):
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(unique_entries)[:5])
                    + "]"
                    + "\nU = "
                    + str(U)
                    + "\nN = "
                    + str(N)
                    + "\nAnnotation = "
                    + annotation
                    + "\nDT+LR    posterior = "
                    + json.dumps(dtandlr_prob).replace('"', "")
                    + "\nptype  posterior = "
                    + json.dumps(ptype_prob).replace('"', "")
                    + "\nprediction_ptype = "
                    + prediction_ptype
                    + "\nprediction_dtandlr = "
                    + prediction_dtandlr
                    + "\n\n"
                )
                append_textfile(
                    "outputs/failure_cases/dtandlr_ptype_" + annotation + ".txt", txt
                )

            if (
                annotation == prediction_ptype
                and prediction_ptype != prediction_dtandlr
            ):
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(unique_entries)[:5])
                    + "]"
                    + "\nU = "
                    + str(U)
                    + "\nN = "
                    + str(N)
                    + "\nAnnotation = "
                    + annotation
                    + "\nDT+LR    posterior = "
                    + json.dumps(dtandlr_prob).replace('"', "")
                    + "\nptype  posterior = "
                    + json.dumps(ptype_prob).replace('"', "")
                    + "\nprediction_ptype = "
                    + prediction_ptype
                    + "\nprediction_dtandlr = "
                    + prediction_dtandlr
                    + "\n\n"
                )
                append_textfile(
                    "outputs/failure_cases/ptype_dtandlr_" + annotation + ".txt", txt
                )


def print_failure_cases_bot_ptype(
    X_tests, trues, all_preds, all_pred_scores, dataset_column_zs, X, z
):
    for t in constants.CANONICAL_TYPES_SORTED:
        if os.path.exists("outputs/failure_cases/bot_ptype_" + t + ".txt"):
            os.remove("outputs/failure_cases/bot_ptype_" + t + ".txt")
        if os.path.exists("outputs/failure_cases/ptype_bot_" + t + ".txt"):
            os.remove("outputs/failure_cases/ptype_bot_" + t + ".txt")

    for i in range(len(X_tests)):
        annotations = trues[i]

        predictions_bot = all_preds[0][i]
        predictions_ptype = all_preds[1][i]

        for (j, (annotation, prediction_bot, prediction_ptype),) in enumerate(
            zip(annotations, predictions_bot, predictions_ptype)
        ):
            bot_scores = all_pred_scores[0][i]
            bot_prob = get_scores("bot", j=j, scores=bot_scores)

            ptype_scores = all_pred_scores[1][i]
            ptype_prob = get_scores("ptype", j=j, scores=ptype_scores)

            z_i = dataset_column_zs[i][j]
            dataset, column, unique_entries = z_i
            index = sum(
                [
                    index_i
                    for index_i, z_i_temp in enumerate(z)
                    if (z_i_temp == z_i).all()
                ]
            )
            features = X[index, :]
            U = int(features[7])
            N = int(features[9])

            if annotation == prediction_bot and prediction_ptype != prediction_bot:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(unique_entries)[:5])
                    + "]"
                    + "\nU = "
                    + str(U)
                    + "\nN = "
                    + str(N)
                    + "\nAnnotation = "
                    + annotation
                    + "\nbot    posterior = "
                    + json.dumps(bot_prob).replace('"', "")
                    + "\nptype  posterior = "
                    + json.dumps(ptype_prob).replace('"', "")
                    + "\n\n"
                )
                append_textfile(
                    "outputs/failure_cases/bot_ptype_" + annotation + ".txt", txt
                )

            if annotation == prediction_ptype and prediction_ptype != prediction_bot:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(unique_entries)[:5])
                    + "]"
                    + "\nU = "
                    + str(U)
                    + "\nN = "
                    + str(N)
                    + "\nAnnotation = "
                    + annotation
                    + "\nbot    posterior = "
                    + json.dumps(bot_prob).replace('"', "")
                    + "\nptype  posterior = "
                    + json.dumps(ptype_prob).replace('"', "")
                    + "\n\n"
                )
                append_textfile(
                    "outputs/failure_cases/ptype_bot_" + annotation + ".txt", txt
                )


def print_failure_cases_metadata(all_preds, trues, all_pred_scores, add_infos):
    for method in ["keyword", "bi_lstm", "bi_lstm_attention"]:
        remove_existing_files(method)
    remove_existing_files("keyword_bi_lstm")
    remove_existing_files("keyword_roberta")
    remove_existing_files("roberta_keyword")
    remove_existing_files("roberta_bi-lstm")
    remove_existing_files("bi_lstm_keyword")
    remove_existing_files("ptype_bi_lstm")
    remove_existing_files("LR_bi-lstm")
    remove_existing_files("kb")

    for i in range(5):
        annotations = trues[i]
        predictions_ptype = all_preds[0][i]
        predictions_lr = all_preds[1][i]
        predictions_keyword = all_preds[3][i]
        predictions_bi_lstm = all_preds[4][i]
        predictions_bi_lstm_attention = all_preds[5][i]
        predictions_roberta = all_preds[6][i]

        for (
            j,
            (
                annotation,
                prediction_ptype,
                prediction_lr,
                prediction_keyword,
                prediction_bi_lstm,
                prediction_bi_lstm_attention,
                prediction_roberta,
            ),
        ) in enumerate(
            zip(
                annotations,
                predictions_ptype,
                predictions_lr,
                predictions_keyword,
                predictions_bi_lstm,
                predictions_bi_lstm_attention,
                predictions_roberta,
            )
        ):
            # ptype_scores = all_pred_scores[0][i]
            # ptype_prob = get_scores("ptype", j=j, scores=ptype_scores)

            # lr_scores = all_pred_scores[1][i]
            # lr_prob = get_scores("LogisticRegression", j=j, scores=lr_scores)

            keyword_scores = all_pred_scores[3][i]
            keyword_prob = get_scores("keyword", j=j, scores=keyword_scores)

            bi_lstm_scores = all_pred_scores[4][i]
            bi_lstm_prob = get_scores("bi-lstm", j=j, scores=bi_lstm_scores)

            bi_lstm_attention_scores = all_pred_scores[5][i]
            bi_lstm_attention_prob = get_scores(
                "bi-lstm-attention", j=j, scores=bi_lstm_attention_scores
            )

            roberta_scores = all_pred_scores[6][i]
            roberta_prob = get_scores("roberta", j=j, scores=roberta_scores)

            dataset, column, entries, seq, _, _ = add_infos[(i, "test")][j]
            # u_ratio_clean = data[j, constants.ADDITIONAL_FEATURES["u_ratio_clean"]]
            # U_clean = data[j, constants.ADDITIONAL_FEATURES["U_clean"]]

            append_to_file(
                prediction_keyword,
                i,
                dataset,
                column,
                entries,
                seq,
                annotation,
                keyword_prob,
                bi_lstm_prob,
                bi_lstm_attention_prob,
                "keyword_",
            )

            append_to_file(
                prediction_bi_lstm,
                i,
                dataset,
                column,
                entries,
                seq,
                annotation,
                keyword_prob,
                bi_lstm_prob,
                bi_lstm_attention_prob,
                "bi_lstm_",
            )

            append_to_file(
                prediction_bi_lstm_attention,
                i,
                dataset,
                column,
                entries,
                seq,
                annotation,
                keyword_prob,
                bi_lstm_prob,
                bi_lstm_attention_prob,
                "bi_lstm_attention_",
            )

            # if annotation != prediction_keyword:
            #     txt = get_text(
            #         i,
            #         dataset,
            #         column,
            #         entries,
            #         seq,
            #         annotation,
            #         keyword_prob,
            #         bi_lstm_prob,
            #         bi_lstm_attention_prob,
            #     )
            #
            #     append_textfile("outputs/keyword_" + annotation + ".txt", txt)

            # if annotation != prediction_bi_lstm:
            #     txt = get_text(
            #         i,
            #         dataset,
            #         column,
            #         entries,
            #         seq,
            #         annotation,
            #         keyword_prob,
            #         bi_lstm_prob,
            #         bi_lstm_attention_prob,
            #     )
            #
            #     append_textfile("outputs/bi_lstm_" + annotation + ".txt", txt)

            # if annotation != prediction_bi_lstm_attention:
            #     txt = get_text(
            #         i,
            #         dataset,
            #         column,
            #         entries,
            #         seq,
            #         annotation,
            #         keyword_prob,
            #         bi_lstm_prob,
            #         bi_lstm_attention_prob,
            #     )
            #
            #     append_textfile("outputs/bi_lstm_attention_" + annotation + ".txt", txt)

            if prediction_keyword == annotation and prediction_bi_lstm != annotation:
                txt = get_text(
                    i,
                    dataset,
                    column,
                    entries,
                    seq,
                    annotation,
                    keyword_prob,
                    bi_lstm_prob,
                    roberta_prob,
                )

                append_textfile(
                    "outputs/failure_cases/keyword_bi_lstm_" + annotation + ".txt", txt
                )

            if prediction_keyword != annotation and prediction_bi_lstm == annotation:
                txt = get_text(
                    i,
                    dataset,
                    column,
                    entries,
                    seq,
                    annotation,
                    keyword_prob,
                    bi_lstm_prob,
                    roberta_prob,
                )

                append_textfile(
                    "outputs/failure_cases/bi_lstm_keyword_" + annotation + ".txt", txt
                )

            if prediction_keyword != annotation and prediction_bi_lstm != annotation:
                txt = get_text(
                    i,
                    dataset,
                    column,
                    entries,
                    seq,
                    annotation,
                    keyword_prob,
                    bi_lstm_prob,
                    roberta_prob,
                )

                append_textfile("outputs/failure_cases/kb_" + annotation + ".txt", txt)

            if prediction_keyword == annotation and prediction_roberta != annotation:
                txt = get_text(
                    i,
                    dataset,
                    column,
                    entries,
                    seq,
                    annotation,
                    keyword_prob,
                    roberta_prob,
                    roberta_prob,
                )

                append_textfile(
                    "outputs/failure_cases/keyword_roberta_" + annotation + ".txt", txt
                )

            if prediction_keyword != annotation and prediction_roberta == annotation:
                txt = get_text(
                    i,
                    dataset,
                    column,
                    entries,
                    seq,
                    annotation,
                    keyword_prob,
                    roberta_prob,
                    bi_lstm_attention_prob,
                )

                append_textfile(
                    "outputs/failure_cases/roberta_keyword_" + annotation + ".txt", txt
                )

            if prediction_bi_lstm != annotation and prediction_roberta == annotation:
                txt = get_text(
                    i,
                    dataset,
                    column,
                    entries,
                    seq,
                    annotation,
                    bi_lstm_prob,
                    roberta_prob,
                    bi_lstm_attention_prob,
                )

                append_textfile(
                    "outputs/failure_cases/roberta_bi-lstm_" + annotation + ".txt", txt
                )


def print_failure_cases_lr(X_tests, all_preds, trues, all_pred_scores, add_infos):
    for t in constants.CANONICAL_TYPES_SORTED:
        if os.path.exists("outputs/failure_cases/LR_" + t + ".txt"):
            os.remove("outputs/failure_cases/LR_" + t + ".txt")

    for i in range(len(X_tests)):
        annotations = trues[i]

        predictions_ptype = all_preds[0][i]
        predictions_LR = all_preds[1][i]
        predictions_DT = all_preds[2][i]

        for (
            j,
            (annotation, prediction_ptype, prediction_LR, prediction_DT),
        ) in enumerate(
            zip(annotations, predictions_ptype, predictions_LR, predictions_DT)
        ):
            ptype_scores = all_pred_scores[0][i]
            # ptype_scores = list(data[j, :2]) + [data[j, 2] + data[j, 3], data[j, 4]]
            ptype_prob = get_scores("ptype", j=j, scores=ptype_scores)

            # lstm_scores = list(data[j, -4:])
            # lstm_prob = get_scores("lstm", scores=lstm_scores)

            LR_scores = all_pred_scores[1][i]
            LR_prob = get_scores("LogisticRegression", j=j, scores=LR_scores)

            DT_scores = all_pred_scores[2][i]
            DT_prob = get_scores("DecisionTree", j=j, scores=DT_scores)

            dataset, column, entries, seq, _, _ = add_infos[i][j]
            # u_ratio_clean = data[j, constants.ADDITIONAL_FEATURES["u_ratio_clean"]]
            # U_clean = data[j, constants.ADDITIONAL_FEATURES["U_clean"]]

            if annotation != prediction_LR:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\n# Entries = "
                    + str(len(entries))
                    + "\n# Unique Entries = "
                    + str(len(np.unique(entries)))
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(entries)[:5])
                    + "]"
                    + "\nMetadata Sequence = "
                    + seq
                    + "Annotation = "
                    + annotation
                    + "\nptype    posterior = "
                    + json.dumps(ptype_prob).replace('"', "")
                    # + "\nlstm    posterior = "
                    # + json.dumps(lstm_prob).replace('"', "")
                    + "\nLog Reg  posterior = "
                    + json.dumps(LR_prob).replace('"', "")
                    + "\nDec Tree posterior = "
                    + json.dumps(DT_prob).replace('"', "")
                    + "\n\n"
                )
                append_textfile("outputs/failure_cases/LR_" + annotation + ".txt", txt)


def print_failure_cases_dt(X_tests, all_preds, trues, all_pred_scores, add_infos):
    for t in constants.CANONICAL_TYPES_SORTED:
        if os.path.exists("outputs/failure_cases/DT_" + t + ".txt"):
            os.remove("outputs/failure_cases/DT_" + t + ".txt")

    for i in range(len(X_tests)):
        data = X_tests[i]
        annotations = trues[i]

        predictions_ptype = all_preds[0][i]
        predictions_LR = all_preds[1][i]
        predictions_DT = all_preds[2][i]

        for (
            j,
            (annotation, prediction_ptype, prediction_LR, prediction_DT),
        ) in enumerate(
            zip(annotations, predictions_ptype, predictions_LR, predictions_DT)
        ):
            ptype_scores = list(data[j, :2]) + [data[j, 2] + data[j, 3], data[j, 4]]
            ptype_prob = get_scores("ptype", scores=ptype_scores)

            lstm_scores = list(data[j, -4:])
            lstm_prob = get_scores("lstm", scores=lstm_scores)

            LR_scores = all_pred_scores[1][i]
            LR_prob = get_scores("LogisticRegression", j=j, scores=LR_scores)

            DT_scores = all_pred_scores[2][i]
            DT_prob = get_scores("DecisionTree", j=j, scores=DT_scores)

            dataset, column, entries, seq, _, _ = add_infos[i][j]
            # u_ratio_clean = data[j, constants.ADDITIONAL_FEATURES["u_ratio_clean"]]
            # U_clean = data[j, constants.ADDITIONAL_FEATURES["U_clean"]]

            if annotation != prediction_DT:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\n# Entries = "
                    + str(len(entries))
                    + "\n# Unique Entries = "
                    + str(len(np.unique(entries)))
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(entries)[:5])
                    + "]"
                    + "\nMetadata Sequence = "
                    + seq
                    + "Annotation = "
                    + annotation
                    + "\nptype    posterior = "
                    + json.dumps(ptype_prob).replace('"', "")
                    + "\nlstm    posterior = "
                    + json.dumps(lstm_prob).replace('"', "")
                    + "\nLog Reg  posterior = "
                    + json.dumps(LR_prob).replace('"', "")
                    + "\nDec Tree posterior = "
                    + json.dumps(DT_prob).replace('"', "")
                    + "\n\n"
                )
                append_textfile("outputs/failure_cases/DT_" + annotation + ".txt", txt)


def print_failure_cases_ptype(X_tests, all_preds, trues, index2data, index2data_tests):
    for i in range(len(X_tests)):
        data = X_tests[i]
        annotations = trues[i]

        predictions_ptype = all_preds[0][i]

        for (j, (annotation, prediction_ptype),) in enumerate(
            zip(annotations, predictions_ptype)
        ):
            temp_index = index2data_tests[i][j]

            ptype_scores = list(data[j, :2]) + [data[j, 2] + data[j, 3], data[j, 4]]
            ptype_prob = get_scores("ptype", scores=ptype_scores)

            dataset, column, entries, seq, mentions = index2data[temp_index]

            if annotation == "string" and prediction_ptype == "categorical":
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nData Entries = ["
                    + ", ".join(entries[:5])
                    + "]"
                    + "\nMetadata Sequence = "
                    + seq
                    + "Annotation = "
                    + annotation
                    + "\nMention = "
                    + ", ".join(mentions)
                    + "\nptype    posterior = "
                    + json.dumps(ptype_prob).replace('"', "")
                    + "\n\n"
                )

                append_textfile("outputs/failure_cases_ptype.txt", txt)


def print_failure_cases_sum(all_preds, trues, all_pred_scores, add_infos, methods):
    remove_existing_files("sum")
    remove_existing_files("sum_LR_LR+KS+R")
    remove_existing_files("sum_LR_LR+R")
    remove_existing_files("sum_LR+R_LR+KS+R")
    remove_existing_files("sum_LR+KS+R_LR+R")
    remove_existing_files("sum_LR+R_LR")
    method_names = [
        "ptype",
        "LR",
        "KS",
        "R",
        "ptype+KS",
        "ptype+R",
        "LR+KS",
        "LR+R",
        "LR+KS+R",
    ]

    for i in range(5):
        annotations = trues[i]
        predictions_ptype = all_preds[0][i]
        predictions_LR = all_preds[1][i]
        predictions_KS = all_preds[2][i]
        predictions_R = all_preds[3][i]
        predictions_ptype_KS = all_preds[4][i]
        predictions_ptype_R = all_preds[5][i]
        predictions_LR_KS = all_preds[6][i]
        predictions_LR_R = all_preds[7][i]
        predictions_LR_KS_R = all_preds[8][i]

        for (
            j,
            (
                annotation,
                prediction_ptype,
                prediction_LR,
                prediction_KS,
                prediction_R,
                prediction_ptype_KS,
                prediction_ptype_R,
                prediction_LR_KS,
                prediction_LR_R,
                prediction_LR_KS_R,
            ),
        ) in enumerate(
            zip(
                annotations,
                predictions_ptype,
                predictions_LR,
                predictions_KS,
                predictions_R,
                predictions_ptype_KS,
                predictions_ptype_R,
                predictions_LR_KS,
                predictions_LR_R,
                predictions_LR_KS_R,
            )
        ):
            probs = {
                method_name: get_scores(
                    method_name, j=j, scores=all_pred_scores[method_index][i]
                )
                for method_index, method_name in enumerate(method_names)
            }

            dataset, column, entries, seq, _, _ = add_infos[(i, "test")][j]
            # u_ratio_clean = data[j, constants.ADDITIONAL_FEATURES["u_ratio_clean"]]
            # U_clean = data[j, constants.ADDITIONAL_FEATURES["U_clean"]]

            if prediction_LR != annotation and prediction_LR_KS_R == annotation:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\n# Entries = "
                    + str(len(entries))
                    + "\n# Unique Entries = "
                    + str(len(np.unique(entries)))
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(entries)[:5])
                    + "]"
                    + "\nMetadata Sequence = "
                    + seq
                    + "Annotation = "
                    + annotation
                    + "\nLR probabilities = "
                    + json.dumps(probs["LR"]).replace('"', "")
                    + "\nLR+R probabilities = "
                    + json.dumps(probs["LR+R"]).replace('"', "")
                    + "\nLR+KS+R probabilities = "
                    + json.dumps(probs["LR+KS+R"]).replace('"', "")
                    + "\n\n"
                )

                append_textfile(
                    "outputs/failure_cases/sum_LR_LR+KS+R_" + annotation + ".txt", txt
                )

            if prediction_LR != annotation and prediction_LR_R == annotation:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\n# Entries = "
                    + str(len(entries))
                    + "\n# Unique Entries = "
                    + str(len(np.unique(entries)))
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(entries)[:5])
                    + "]"
                    + "\nMetadata Sequence = "
                    + seq
                    + "Annotation = "
                    + annotation
                    + "\nLR probabilities = "
                    + json.dumps(probs["LR"]).replace('"', "")
                    + "\nLR+R probabilities = "
                    + json.dumps(probs["LR+R"]).replace('"', "")
                    + "\nLR+KS+R probabilities = "
                    + json.dumps(probs["LR+KS+R"]).replace('"', "")
                    + "\n\n"
                )

                append_textfile(
                    "outputs/failure_cases/sum_LR_LR+R_" + annotation + ".txt", txt,
                )

            if prediction_LR == annotation and prediction_LR_R != annotation:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\n# Entries = "
                    + str(len(entries))
                    + "\n# Unique Entries = "
                    + str(len(np.unique(entries)))
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(entries)[:5])
                    + "]"
                    + "\nMetadata Sequence = "
                    + seq
                    + "Annotation = "
                    + annotation
                    + "\nLR probabilities = "
                    + json.dumps(probs["LR"]).replace('"', "")
                    + "\nR probabilities = "
                    + json.dumps(probs["R"]).replace('"', "")
                    + "\nLR+R probabilities = "
                    + json.dumps(probs["LR+R"]).replace('"', "")
                    + "\nLR+KS+R probabilities = "
                    + json.dumps(probs["LR+KS+R"]).replace('"', "")
                    + "\n\n"
                )

                append_textfile(
                    "outputs/failure_cases/sum_LR+R_LR_" + annotation + ".txt", txt,
                )

            if prediction_LR_R != annotation and prediction_LR_KS_R == annotation:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\n# Entries = "
                    + str(len(entries))
                    + "\n# Unique Entries = "
                    + str(len(np.unique(entries)))
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(entries)[:5])
                    + "]"
                    + "\nMetadata Sequence = "
                    + seq
                    + "Annotation = "
                    + annotation
                    + "\nLR probabilities = "
                    + json.dumps(probs["LR"]).replace('"', "")
                    + "\nR probabilities = "
                    + json.dumps(probs["R"]).replace('"', "")
                    + "\nLR+R probabilities = "
                    + json.dumps(probs["LR+R"]).replace('"', "")
                    + "\nKS probabilities = "
                    + json.dumps(probs["KS"]).replace('"', "")
                    + "\nLR+KS+R probabilities = "
                    + json.dumps(probs["LR+KS+R"]).replace('"', "")
                    + "\n\n"
                )

                append_textfile(
                    "outputs/failure_cases/sum_LR+R_LR+KS+R_" + annotation + ".txt",
                    txt,
                )

            if prediction_LR_R == annotation and prediction_LR_KS_R != annotation:
                txt = (
                    "Dataset = "
                    + dataset
                    + "\nColumn = "
                    + column
                    + "\nFold = "
                    + str(i)
                    + "\n# Entries = "
                    + str(len(entries))
                    + "\n# Unique Entries = "
                    + str(len(np.unique(entries)))
                    + "\nSome Unique Data Entries = ["
                    + ", ".join(np.unique(entries)[:5])
                    + "]"
                    + "\nMetadata Sequence = "
                    + seq
                    + "Annotation = "
                    + annotation
                    + "\nLR probabilities = "
                    + json.dumps(probs["LR"]).replace('"', "")
                    + "\nR probabilities = "
                    + json.dumps(probs["R"]).replace('"', "")
                    + "\nLR+R probabilities = "
                    + json.dumps(probs["LR+R"]).replace('"', "")
                    + "\nKS probabilities = "
                    + json.dumps(probs["KS"]).replace('"', "")
                    + "\nLR+KS+R probabilities = "
                    + json.dumps(probs["LR+KS+R"]).replace('"', "")
                    + "\n\n"
                )

                append_textfile(
                    "outputs/failure_cases/sum_LR+KS+R_LR+R_" + annotation + ".txt",
                    txt,
                )
