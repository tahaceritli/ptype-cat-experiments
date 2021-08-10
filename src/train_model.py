from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from src import constants

import joblib
import numpy as np


def reshape_pred_scores(pred_scores):
    y_probs = ()
    for y_prob in pred_scores:
        n = int(len(y_prob) / 4)
        y_prob = np.array(y_prob).reshape(n, 4)
        y_probs += (y_prob,)
    y_probs = np.vstack(y_probs)
    return y_probs


def get_pred_scores_combined(p, y_tests):
    y_tests_hat = []
    pred_scores = []
    for i in range(constants.FOLD):
        if i == 0:
            y_test_hat = p[: len(y_tests[i]), :]
            last_index = len(y_tests[i])
        else:
            y_test_hat = p[last_index : last_index + len(y_tests[i]), :]
            last_index = last_index + len(y_tests[i])
        pred_scores.append(list(y_test_hat.flatten()))
        y_test_hat = np.argmax(y_test_hat, axis=1)
        y_test_hat = [constants.CANONICAL_TYPES_SORTED[index] for index in y_test_hat]
        y_tests_hat.append(y_test_hat)

    return pred_scores, y_tests_hat


robust_scaler = RobustScaler()

# load inputs
X = np.load("inputs/X.npy")
y = np.load("inputs/y.npy")
z = np.load("inputs/z.npy", allow_pickle=True)

# scale features
features = ["u_ratio", "u_ratio_clean", "U", "U_clean"]
features_indices = [constants.ADDITIONAL_FEATURES[feat] for feat in features]
# magic numbers below
X_feats = X[:, list(range(4)) + features_indices]
X_feats[:, [6, 7]] = robust_scaler.fit_transform(X_feats[:, [6, 7]])

# run classifier
clf = LogisticRegression(multi_class="multinomial", max_iter=10000, penalty="l2").fit(
    X_feats, y
)
y_hat = clf.predict(X_feats)

# save the model and scaler
joblib.dump(robust_scaler, "outputs/models/robust_scaler.pkl")
joblib.dump(clf, "outputs/models/LR.sav")
