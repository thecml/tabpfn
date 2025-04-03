import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# Generate synthetic survival data
np.random.seed(42)
n_samples = 500

X = np.random.normal(size=(n_samples, 5))  # 5 features
event_times = np.random.exponential(scale=10, size=n_samples)  # Survival times
event_observed = np.random.binomial(1, 0.7, size=n_samples)  # Censoring indicator (1=event, 0=censored)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
df["time"] = event_times
df["event"] = event_observed

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["time", "event"]), df[["time", "event"]], test_size=0.2, random_state=42
)

# Convert time to -log(time) for Cox model
y_train_transformed = -np.log(y_train["time"])
y_test_transformed = -np.log(y_test["time"])

# Prepare DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train_transformed)
dtest = xgb.DMatrix(X_test, label=y_test_transformed)

# Train XGBoost model with Cox regression objective
params = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "eta": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "seed": 42
}

model = xgb.train(params, dtrain, num_boost_round=100)

# Predict risk scores (higher = higher risk)
risk_scores = model.predict(dtest)

# Evaluate using Concordance Index (C-Index)
c_index = concordance_index(y_test["time"], -risk_scores, y_test["event"])
print(f"Concordance Index: {c_index:.4f}")