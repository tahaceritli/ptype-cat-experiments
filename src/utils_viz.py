import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab as P

from itertools import combinations
from mpltools import special
from sklearn.metrics import (
    # auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

from src import constants
from src.utils_data import sort_array, sort_array_metadata

# from src.utils_eval import flatten


def _blob(x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    P.fill(xcorners, ycorners, colour, edgecolor=colour)


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=["black", "white"],
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    colorbar=None,
    x_label=None,
    y_label=None,
    vmin=-1.0,
    vmax=1.0,
    rotation=30,
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca()

    # Plot the heatmap
    im = ax.imshow(
        data, cmap=plt.cm.gray_r, vmax=vmax, vmin=vmin, aspect="auto", **kwargs
    )
    if colorbar is not None:
        fig.colorbar(im, ax=ax)

    # # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if rotation == 0.0:
        ax.set_xticklabels(col_labels, fontsize=15)
    else:
        ax.set_xticklabels(
            col_labels,
            fontsize=15,
            rotation=rotation,
            ha="left",
            rotation_mode="anchor",
        )
    ax.set_yticklabels(row_labels, fontsize=15)

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=20)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=20)
    ax.xaxis.set_label_position("top")

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(),

    # Turn spines off and create white grid.
    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(True, which="minor")
    # ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def hinton(W, labels, filepath, maxWeight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    """
    reenable = False
    if P.isinteractive():
        P.ioff()
    # P.figure(figsize=(8, 8))
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2 ** np.ceil(np.log(np.max(np.abs(W))) / np.log(2))

    P.fill(np.array([0, width, width, 0]), np.array([0, 0, height, height]), "gray")
    # P.axis("off")
    # P.axis("equal")
    for x in range(width):
        for y in range(height):
            _x = x + 1
            _y = y + 1
            w = W[y, x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1, w / maxWeight), "white")
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1, -w / maxWeight), "black")
    if reenable:
        P.ion()

    # We want to show all ticks...

    P.xticks(
        [x + 0.5 for x in range(0, len(labels))],
        labels,
        fontsize=18,
        rotation=30,
        ha="left",
        rotation_mode="anchor",
    )
    P.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    P.yticks([x + 0.5 for x in range(0, len(labels))], reversed(labels), fontsize=18)
    P.tight_layout()
    # P.show()
    #
    P.savefig(filepath, dpi=400, bbox_inches="tight")


def hinton_special(W, labels, filepath):
    plt.figure()
    special.hinton(W)
    plt.xticks(
        range(len(labels)),
        labels,
        fontsize=20,
        ha="left",
        rotation=30,
        rotation_mode="anchor",
    )
    plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.yticks(range(len(labels)), labels, fontsize=20)
    plt.xlabel("True Class", fontsize=24)
    plt.ylabel("Predicted Class", fontsize=24)
    plt.gca().xaxis.set_label_position("top")
    plt.tight_layout()
    plt.savefig(filepath, dpi=400, bbox_inches="tight")


def hinton_rectangular(W, x_labels, y_labels, filepath, x_rotation=30, maxWeight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    """
    reenable = False
    if P.isinteractive():
        P.ioff()
    P.figure(figsize=(8, 8))
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2 ** np.ceil(np.log(np.max(np.abs(W))) / np.log(2))

    P.fill(np.array([0, width, width, 0]), np.array([0, 0, height, height]), "gray")
    # P.axis("off")
    # P.axis("equal")
    for x in range(width):
        for y in range(height):
            _x = x + 1
            _y = y + 1
            w = W[y, x]
            if w > 0:
                # _blob(_x, height - _y, min(1, w / maxWeight), "white")
                _blob(_x - 0.5, height - _y + 0.5, min(1, w / maxWeight), "white")
            elif w < 0:
                # _blob(_x, height - _y, min(1, -w / maxWeight), "black")
                _blob(_x - 0.5, height - _y + 0.5, min(1, -w / maxWeight), "black")
    if reenable:
        P.ion()

    # # We want to show all ticks...

    P.xticks(
        [x + 0.5 for x in range(0, len(x_labels))],
        x_labels,
        fontsize=24,
        rotation=x_rotation,
        ha="left",
        rotation_mode="anchor",
    )
    P.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    P.yticks(
        [x + 0.5 for x in range(0, len(y_labels))], reversed(y_labels), fontsize=24
    )

    P.tight_layout()
    P.savefig(filepath, dpi=400, bbox_inches="tight")


def plot_confusion_matrix(y_tests, y_tests_hat, labels, filepath):
    C = len(labels)
    confusion_matrix = np.zeros((C, C), dtype=int)
    for _annotations, _predictions in zip(y_tests, y_tests_hat):
        for annotation, prediction in zip(_annotations, _predictions):

            if prediction != "any":
                confusion_matrix[
                    labels.index(prediction), labels.index(annotation),
                ] += 1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5),)
    fig.tight_layout()

    im = heatmap(
        confusion_matrix,
        labels,
        labels,
        x_label="True Class",
        y_label="Predicted Class",
        ax=ax,
        vmin=0,
        vmax=0,
    )
    annotate_heatmap(im, valfmt="{x:d}", size=20, textcolors=["black", "white"])
    fig.tight_layout()

    plt.savefig(filepath, dpi=300, bbox_inches="tight")


def plot_hinton(y_tests, y_tests_hat, labels, filepath):
    C = len(labels)
    confusion_matrix = np.zeros((C, C), dtype=int)

    for _annotations, _predictions in zip(y_tests, y_tests_hat):
        for annotation, prediction in zip(_annotations, _predictions):

            if prediction != "any":
                confusion_matrix[
                    labels.index(prediction), labels.index(annotation),
                ] += 1
    normalized_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0)
    hinton_special(normalized_confusion_matrix, labels, filepath=filepath)


def plot_all_confusion_matrices(y_tests, all_y_tests_hat, methods, filepath):
    C = len(constants.CANONICAL_TYPES_SORTED)
    fig, axs = plt.subplots(
        nrows=1, ncols=7, figsize=(3 * len(methods), 2 * len(methods)),
    )
    fig.tight_layout()
    for method, y_tests_hat, ax in zip(methods, all_y_tests_hat, axs.reshape(-1)):
        ax.autoscale(enable=True)

        confusion_matrix = np.zeros((C, C), dtype=int)
        for _annotations, _predictions in zip(y_tests, y_tests_hat):
            for annotation, prediction in zip(_annotations, _predictions):
                if prediction != "any":
                    confusion_matrix[
                        constants.CANONICAL_TYPES_SORTED.index(prediction),
                        constants.CANONICAL_TYPES_SORTED.index(annotation),
                    ] += 1

        im = heatmap(
            confusion_matrix,
            constants.CANONICAL_TYPES_SORTED,
            constants.CANONICAL_TYPES_SORTED,
            x_label="True Class",
            y_label="Predicted Class",
            ax=ax,
            vmin=0,
            vmax=0,
        )
        annotate_heatmap(im, valfmt="{x:d}", size=20, textcolors=["black", "white"])
        ax.text(
            x=0.75 + (12 - len(method)) / 20,
            y=3.75,
            s=constants.method_name_abbs[method],
            fontsize=20,
        )
    fig.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")


def plot_all_confusion_matrices_per_fold(y_tests, all_y_tests_hat, methods, filepath):
    C = len(constants.CANONICAL_TYPES_SORTED)
    for method, y_tests_hat in zip(methods, all_y_tests_hat):
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25, 5),)
        fig.tight_layout()
        for i, (_annotations, _predictions) in enumerate(zip(y_tests, y_tests_hat)):
            confusion_matrix = np.zeros((C, C), dtype=int)
            for annotation, prediction in zip(_annotations, _predictions):
                if prediction != "any":
                    confusion_matrix[
                        constants.CANONICAL_TYPES_SORTED.index(prediction),
                        constants.CANONICAL_TYPES_SORTED.index(annotation),
                    ] += 1
            ax = axs[i]
            im = heatmap(
                confusion_matrix,
                constants.CANONICAL_TYPES_SORTED,
                constants.CANONICAL_TYPES_SORTED,
                x_label="True Class",
                y_label="Predicted Class",
                ax=ax,
                vmin=0,
                vmax=0,
            )
            annotate_heatmap(im, valfmt="{x:d}", size=20, textcolors=["black", "white"])
            ax.text(x=0.75 + (12 - len(method)) / 20, y=3.75, s=method, fontsize=20)
        fig.tight_layout()
        plt.savefig(
            filepath.split(".png")[0] + "_" + method + ".png",
            dpi=300,
            bbox_inches="tight",
        )


def plot_all_confusion_matrices_per_class(
    y_tests, predictions, filepath, method_pairs=None
):
    methods = list(predictions.keys())
    types = constants.DATA2PANDAS_TYPES_SORTED
    C = len(types)
    if method_pairs is None:
        method_pairs = list(combinations(methods, 2))

    for i, (method_1, method_2) in enumerate(method_pairs):
        fig, axs = plt.subplots(nrows=1, ncols=C, figsize=(3 * C, 3))
        fig.tight_layout()

        y_tests_hat_1 = predictions[method_1]["y_hat"]
        y_tests_hat_2 = predictions[method_2]["y_hat"]

        plt.figtext(
            0.5, 1.05, method_2, ha="center", va="top", fontsize=20, color="k",
        )

        plt.figtext(
            0.0,
            0.4,
            method_1,
            ha="center",
            va="center",
            fontsize=20,
            rotation=90,
            color="k",
        )

        for j, t in enumerate(types):
            confusion_matrix = np.zeros((2, 2), dtype=int)
            for predictions_1, predictions_2, annotations in zip(
                y_tests_hat_1, y_tests_hat_2, y_tests
            ):
                for prediction_1, prediction_2, annotation in zip(
                    predictions_1, predictions_2, annotations
                ):
                    if annotation == t:
                        confusion_matrix[
                            int(prediction_1 == annotation),
                            int(prediction_2 == annotation),
                        ] += 1

            ax = axs[j]
            ax.autoscale(enable=True)
            im = heatmap(
                confusion_matrix,
                ["False", "True"],
                ["False", "True"],
                # x_label=method_2,
                # y_label=method_1,
                ax=ax,
                vmin=0,
                vmax=0,
                rotation=0.0,
            )
            annotate_heatmap(im, valfmt="{x:d}", size=20, textcolors=["black", "white"])
            ax.set_title(t.capitalize(), fontdict={"fontsize": 17}, pad=-20)
            # ax.text(x=0.75 + (12 - len(method)) / 20, y=3.75, s=method, fontsize=20)

        fig.tight_layout()
        plt.savefig(
            filepath + method_1 + "_" + method_2 + ".png", dpi=300, bbox_inches="tight"
        )


def plot_all_confusion_matrices_per_class_fold(
    y_tests, all_y_tests_hat, methods, filepath, method_pairs=None
):
    C = len(constants.CANONICAL_TYPES_SORTED)
    if method_pairs is None:
        method_pairs = list(combinations(methods, 2))

    # N = len(method_pairs)

    for i, (method_1, method_2) in enumerate(method_pairs):
        fig, axs = plt.subplots(nrows=5, ncols=C, figsize=(3 * 5, 3 * 4))
        fig.tight_layout()

        method_1_index = methods.index(method_1)
        method_2_index = methods.index(method_2)

        y_tests_hat_1 = all_y_tests_hat[method_1_index]
        y_tests_hat_2 = all_y_tests_hat[method_2_index]

        plt.figtext(
            0.5, 1.05, method_2, ha="center", va="top", fontsize=20, color="k",
        )

        plt.figtext(
            0.0,
            0.4,
            method_1,
            ha="center",
            va="center",
            fontsize=20,
            rotation=90,
            color="k",
        )

        for j, t in enumerate(constants.CANONICAL_TYPES_SORTED):

            for k, (predictions_1, predictions_2, annotations) in enumerate(
                zip(y_tests_hat_1, y_tests_hat_2, y_tests)
            ):
                confusion_matrix = np.zeros((2, 2), dtype=int)
                for prediction_1, prediction_2, annotation in zip(
                    predictions_1, predictions_2, annotations
                ):
                    if annotation == t:
                        confusion_matrix[
                            int(prediction_1 == annotation),
                            int(prediction_2 == annotation),
                        ] += 1
                ax = axs[k, j]
                ax.autoscale(enable=True)
                im = heatmap(
                    confusion_matrix,
                    ["False", "True"],
                    ["False", "True"],
                    # x_label=method_2,
                    # y_label=method_1,
                    ax=ax,
                    vmin=0,
                    vmax=0,
                    rotation=0.0,
                )
                annotate_heatmap(
                    im, valfmt="{x:d}", size=20, textcolors=["black", "white"]
                )
                ax.set_title(t.capitalize(), fontdict={"fontsize": 17}, pad=-20)
            # ax.text(x=0.75 + (12 - len(method)) / 20, y=3.75, s=method, fontsize=20)

        fig.tight_layout()
        plt.savefig(
            filepath + method_1 + method_2 + ".png", dpi=300, bbox_inches="tight"
        )


# def plot_all_confusion_matrices_per_fold(y_tests, all_y_tests_hat, methods, filepath):
#     C = len(constants.CANONICAL_TYPES_SORTED)
#     # N = len(method_pairs)
#
#     for i, method in enumerate(methods):
#         fig, axs = plt.subplots(nrows=1, ncols=C, figsize=(3 * C, 3))
#         fig.tight_layout()
#
#         method_1_index = methods.index(method_1)
#         method_2_index = methods.index(method_2)
#
#         y_tests_hat_1 = all_y_tests_hat[method_1_index]
#         y_tests_hat_2 = all_y_tests_hat[method_2_index]
#
#         plt.figtext(
#             0.5, 1.05, method_2, ha="center", va="top", fontsize=20, color="k",
#         )
#
#         plt.figtext(
#             0.0,
#             0.4,
#             method_1,
#             ha="center",
#             va="center",
#             fontsize=20,
#             rotation=90,
#             color="k",
#         )
#
#         for j, t in enumerate(constants.CANONICAL_TYPES_SORTED):
#             confusion_matrix = np.zeros((2, 2), dtype=int)
#             for predictions_1, predictions_2, annotations in zip(
#                 y_tests_hat_1, y_tests_hat_2, y_tests
#             ):
#                 for prediction_1, prediction_2, annotation in zip(
#                     predictions_1, predictions_2, annotations
#                 ):
#                     if annotation == t:
#                         confusion_matrix[
#                             int(prediction_1 == annotation),
#                             int(prediction_2 == annotation),
#                         ] += 1
#             ax = axs[j]
#             ax.autoscale(enable=True)
#             im = heatmap(
#                 confusion_matrix,
#                 ["False", "True"],
#                 ["False", "True"],
#                 # x_label=method_2,
#                 # y_label=method_1,
#                 ax=ax,
#                 vmin=0,
#                 vmax=0,
#                 rotation=0.0,
#             )
#             annotate_heatmap(im, valfmt="{x:d}", size=20, textcolors=["black", "white"])
#             ax.set_title(t.capitalize(), fontdict={"fontsize": 17}, pad=-20)
#             # ax.text(x=0.75 + (12 - len(method)) / 20, y=3.75, s=method, fontsize=20)
#
#         fig.tight_layout()
#         plt.savefig(
#             filepath + method_1 + method_2 + ".png", dpi=300, bbox_inches="tight"
#         )


def plot_avg_precisions(
    avg_precisions, filepath="outputs/figures/all_avg_precisions.png"
):
    plt.figure()
    colors = {
        "ptype": "o",
        "keyword-search": "r",
        "bi-lstm": "g",
        "bi-lstm-attention": "b",
    }
    for method in avg_precisions:
        alpha_values = list(avg_precisions[method].keys())
        avg_precision_values = list(avg_precisions[method].values())

        if "LogisticRegression" in method:
            linestyle = "-"
        else:
            linestyle = "--"

        plt.plot(
            alpha_values,
            avg_precision_values,
            label=method[0] + "+" + method[1],
            linestyle=linestyle,
            color=colors[method[1]],
        )

    plt.xlabel("alpha")
    plt.ylabel("Average Precision")
    plt.xlim([-20, 20])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right", fontsize=8)
    # plt.title("Micro-averaged over folds")
    plt.savefig(filepath, dpi=400, bbox_inches="tight")


def plot_all_pr_curves(
    all_true_scores,
    all_pred_scores,
    methods=["Decision Tree"],
    filepath="outputs/figures/all_pr_curves.png",
    title="Micro-averaged over folds",
):
    plt.figure()
    for method, pred_scores, true_scores in zip(
        methods, all_pred_scores, all_true_scores
    ):
        if "+" in method:
            label = "+".join(method.split("+"))
        else:
            if "OpenML" in method:
                label = "OpenML"
            elif "Weka" in method:
                label = "Weka"
            else:
                label = method

        y_reals = []
        y_probs = []
        for y_real, y_prob, in zip(true_scores, pred_scores):
            y_reals = y_reals + y_real
            y_probs = y_probs + y_prob

        precision_mean, recall_mean, _ = precision_recall_curve(y_reals, y_probs)
        plt.plot(
            recall_mean,
            precision_mean,
            label=label
            + " (AP "
            + str(round(average_precision_score(y_reals, y_probs, average="micro"), 2))
            + ")",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower left", fontsize=12)
    if title is not None:
        plt.title(title)
    plt.savefig(filepath, dpi=400, bbox_inches="tight")


def plot_all_roc_curves(
    all_true_scores,
    all_pred_scores,
    methods=["Decision Tree"],
    filepath="outputs/figures/all_pr_curves.png",
    title="Micro-averaged over folds",
):
    plt.figure()
    for method, pred_scores, true_scores in zip(
        methods, all_pred_scores, all_true_scores
    ):
        print(method)
        if "+" in method:
            label = "+".join(method.split("+"))
        else:
            if method == "bot":
                label = "Bot"
            elif method == "bi-lstm":
                label = "Bi-LSTM"
            elif method == "roberta":
                label = "RoBERTa"
            elif method == "LogisticRegression":
                label = "LR"
            elif method == "DecisionTree":
                label = "DT"
            elif method == "keyword-search":
                label = "KS"
            else:
                label = method

        y_reals = []
        y_probs = []
        for y_real, y_prob, in zip(true_scores, pred_scores):
            y_reals = y_reals + y_real
            y_probs = y_probs + y_prob
        fpr_rate, tpr_rate, _ = roc_curve(y_reals, y_probs)
        plt.plot(
            fpr_rate,
            tpr_rate,
            label=label
            + " (AUC "
            + str(round(roc_auc_score(y_reals, y_probs, average="micro"), 2))
            + ")",
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(fontsize=8)
    # plt.legend(loc="lower right", fontsize=8)
    if title is not None:
        plt.title(title)
    plt.savefig(filepath, dpi=400, bbox_inches="tight")


def plot_matrix(
    X,
    labels,
    filepath,
    x_labels=None,
    figsize=(12, 4),
    interpolation="none",
    vmin=0,
    vmax=1.0,
    aspect="auto",
    cmap=plt.cm.gray_r,
    xlabel="data columns",
    ylabel="features",
):
    plt.figure(figsize=figsize)
    plt.imshow(
        X.T,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        cmap=cmap,
    )
    plt.colorbar()
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    if x_labels is not None:
        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.savefig(filepath, dpi=400, bbox_inches="tight")


def plot_pr_curve(true_scores, pred_scores, filepath, method="Decision Tree"):
    y_reals = []
    y_probs = []
    for y_real, y_prob, in zip(true_scores, pred_scores):
        y_reals = y_reals + y_real
        y_probs = y_probs + y_prob

    precision_mean, recall_mean, _ = precision_recall_curve(y_reals, y_probs)
    plt.plot(
        recall_mean,
        precision_mean,
        label=method
        + " (AP "
        + str(round(average_precision_score(y_reals, y_probs, average="micro"), 2))
        + ")",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right", fontsize=8)
    plt.title("Micro-averaged over folds")
    plt.savefig(filepath, dpi=400, bbox_inches="tight")


def plot_matrix_per_type(X, labels, method, filepath, y=None, metadata=None):
    if method == "ptype":
        ts = [0, 1, 2, 3, 4]
        p_max = X[:, :5].argmax(axis=1)
    else:
        ts = constants.CANONICAL_TYPES_SORTED

    fig, axs = plt.subplots(len(ts), figsize=(16, 16), sharey=True)
    fig.tight_layout()

    for ax, t in zip(axs, ts):
        if method == "ptype":
            t_indices = np.where(p_max == t)[0]
            title = "predicted ptype type:" + constants.PTYPE_TYPES_SORTED[t]
        else:
            t_indices = np.where(y == t)[0]
            title = "ground truth:" + t

        X_t = X[t_indices, :]
        if metadata is None:
            X_t_sorted = sort_array(X_t)
        else:
            X_t_sorted = sort_array_metadata(X_t)

        subplot_matrix(fig, ax, X_t_sorted, labels, title)

    fig.tight_layout(pad=3.0)
    plt.savefig(filepath, dpi=400, bbox_inches="tight")


def plot_matrix_per_canonical_type(X, y, labels, filepath):
    fig, axs = plt.subplots(4, figsize=(12, 12), sharey=True)
    fig.tight_layout()

    for ax, t in zip(axs, ["categorical", "date", "numeric", "string"]):
        t_indices = np.where(y == t)[0]
        X_t = X[t_indices, :]

        X_t_sorted = sort_array(X_t)

        subplot_matrix(fig, ax, X_t_sorted, labels, t)

    fig.tight_layout(pad=3.0)
    plt.savefig(
        filepath, dpi=400, bbox_inches="tight",
    )


def plot_matrix_per_ptype_type(X, labels, filepath):
    fig, axs = plt.subplots(5, figsize=(12, 12), sharey=True)
    fig.tight_layout()
    p_max = X[:, :5].argmax(axis=1)
    for ax, t in zip(axs, [0, 1, 2, 3, 4]):
        t_indices = np.where(p_max == t)[0]
        X_t = X[t_indices, :]

        X_t_sorted = sort_array(X_t)

        subplot_matrix(fig, ax, X_t_sorted, labels, constants.PTYPE_TYPES_SORTED[t])

    fig.tight_layout(pad=3.0)
    plt.savefig(
        filepath, dpi=400, bbox_inches="tight",
    )


def get_names(feature):
    feature_index = constants.ADDITIONAL_FEATURES[feature]
    feature_label = feature
    # feature_label = (
    #     feature + " (log)" if feature in ["U", "U_clean", "N", "N_clean"] else feature
    # )

    return feature_index, feature_label


def plot_features_per_ptype_type(X, filepath, x_feat="U", y_feat="u_ratio", D=5):
    p_max_indices = np.argmax(X[:, :D], axis=1)
    x_index, x_label = get_names(x_feat)
    y_index, y_label = get_names(y_feat)

    fig, axs = plt.subplots(
        nrows=len(constants.PTYPE_TYPES_SORTED),
        ncols=1,
        sharex=True,
        sharey=True,
        figsize=(6, 6),
    )

    for i, ptype_type in enumerate(constants.PTYPE_TYPES_SORTED):
        ax = axs[i]
        idx = np.where(p_max_indices == i)[0]
        X_x = np.log(X[idx, x_index]) if "log" in x_label else X[idx, x_index]
        X_y = np.log(X[idx, y_index]) if "log" in y_label else X[idx, y_index]
        ax.scatter(
            X_x,
            X_y,
            c=constants.PLOT_COLORS[ptype_type],
            label=ptype_type,
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

        ax.set_title(ptype_type)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    plt.tight_layout()
    fig.savefig(filepath, dpi=400)


def plot_features_per_canonical_type(X, y, filepath, x_feat="U", y_feat="u_ratio"):
    x_index, x_label = get_names(x_feat)
    y_index, y_label = get_names(y_feat)

    fig, axs = plt.subplots(
        nrows=len(constants.CANONICAL_TYPES_SORTED),
        ncols=1,
        sharex=True,
        sharey=True,
        figsize=(6, 6),
    )

    for ax, canonical_type in zip(axs, constants.CANONICAL_TYPES_SORTED):
        idx = np.where(y == canonical_type)[0]
        X_x = np.log(X[idx, x_index]) if "log" in x_label else X[idx, x_index]
        X_y = np.log(X[idx, y_index]) if "log" in y_label else X[idx, y_index]
        ax.scatter(
            X_x,
            X_y,
            c=constants.PLOT_COLORS[canonical_type],
            label=canonical_type,
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

        ax.set_title(canonical_type)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    plt.tight_layout()
    fig.savefig(filepath, dpi=400)


def subplot_matrix(fig, ax, X, labels, title):
    im = ax.imshow(
        X.T, interpolation="none", vmin=0, vmax=1, aspect="auto", cmap=plt.cm.gray_r,
    )
    fig.colorbar(im, ax=ax, cmap=plt.cm.gray_r, orientation="vertical")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=13)

    ax.set_title(title, fontsize=17)
    ax.set_xlabel("data columns", fontsize=17)
    ax.set_ylabel("features", fontsize=17)
