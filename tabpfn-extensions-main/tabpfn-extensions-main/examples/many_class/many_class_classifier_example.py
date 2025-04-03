#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
"""ManyClassClassifier: TabPFN extension for handling classification with many classes.

This module provides a classifier that overcomes TabPFN's limitation on the number of
classes (typically 10) by using a meta-classifier approach. It works by breaking down
multi-class problems into multiple sub-problems, each within TabPFN's class limit.

WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
The ManyClassClassifier creates multiple TabPFN models for handling many classes.
"""

import numpy as np
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.many_class import ManyClassClassifier

from sklearn.utils import shuffle

# Create synthetic dataset with 20 classes
n_classes = 20
X, y = make_multilabel_classification(n_samples=100,
                                      n_features=10,
                                      n_classes=n_classes,
                                      n_labels=2,
                                      random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

# Initialize base TabPFN classifier
clf_base = TabPFNClassifier()

# Initialize ManyClassClassifier with TabPFN as base estimator
clf = ManyClassClassifier(
    estimator=clf_base,
    alphabet_size=10,  # TabPFN supports up to 10 classes by default
    n_estimators_redundancy=4,  # Increase redundancy for better stability
    random_state=42,
)

# Fit the classifier
clf.fit(X_train, y_train)

# Make predictions
prediction_probabilities = clf.predict_proba(X_test)
predictions = np.argmax(prediction_probabilities, axis=1)

# Evaluate performance
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities, multi_class="ovr"))
print("Accuracy", accuracy_score(y_test, predictions))
