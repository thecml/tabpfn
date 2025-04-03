#  Copyright (c) Prior Labs GmbH 2025.
"""Example of using TabPFN for binary classification.

This example demonstrates how to use TabPFNClassifier on a binary classification task
using the breast cancer dataset from scikit-learn.
"""

from typing import List, Tuple, Union
import numpy as np
from sklearn.datasets import load_breast_cancer, make_multilabel_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from tabpfn import TabPFNClassifier

from sksurv.datasets import load_whas500
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored

import pandas as pd
import torch

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def encode_survival(
        time: Union[float, int, NumericArrayLike],
        event: Union[int, bool, NumericArrayLike],
        bins: NumericArrayLike
) -> torch.Tensor:
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    # TODO this should handle arrays and (CUDA) tensors
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                    dtype=torch.float,
                    device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()

def reformat_survival(
        dataset: pd.DataFrame,
        time_bins: NumericArrayLike,
        dtype: torch.dtype
) -> (torch.Tensor, torch.Tensor):
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=dtype)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y

# Load data
#X, y = load_breast_cancer(return_X_y=True)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
"""
n_classes = 20
X, y = make_multilabel_classification(n_samples=1000,
                                      n_features=10,
                                      n_classes=n_classes,
                                      n_labels=2,
                                      random_state=42)
"""

X, y = load_whas500()
X = X.astype(float)

num_bins = 10
n_classes = num_bins+1
time_bins = np.unique(np.quantile(y['lenfol'], np.linspace(0, 1, num_bins)))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

data_train = X_train.copy()
data_train["time"] = pd.Series(y_train['lenfol'])
data_train["event"] = pd.Series(y_train['fstat']).astype(int)
data_test = X_test.copy()
data_test["time"] = pd.Series(y_test['lenfol'])
data_test["event"] = pd.Series(y_test['fstat']).astype(int)

dtype = torch.float32
X_train, y_train = reformat_survival(data_train, time_bins, dtype)
X_test, y_test = reformat_survival(data_test, time_bins, dtype)

base_clf = TabPFNClassifier()
clf = MultiOutputClassifier(base_clf)

clf.fit(X_train, y_train)

# Function to get probabilities safely
def get_positive_proba(estimator, X):
    proba = estimator.predict_proba(X)
    return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0] # Use [:, 0] if only one column

# Predict probabilities safely
prediction_probabilities = np.stack([get_positive_proba(est, X_test) for est in clf.estimators_], axis=1)

pmf = torch.tensor(prediction_probabilities)
survival_curve = torch.cumprod(1 - pmf, dim=1)

# Compute ROC AUC per class and average
roc_auc_scores = [roc_auc_score(y_test[:, i], prediction_probabilities[:, i]) for i in range(n_classes)]
print(f"Mean ROC AUC: {np.mean(roc_auc_scores):.4f}")

# Predict labels
predictions = clf.predict(X_test)

# Compute accuracy per class and average
accuracies = [accuracy_score(y_test[:, i], predictions[:, i]) for i in range(n_classes)]
print(f"Mean Accuracy: {np.mean(accuracies):.4f}")

"""
# Initialize a classifier
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
"""