import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
df = pd.read_csv("maternal_health_dataset.csv")
df["Smoking_Status"] = df["Smoking_Status"].map({"Yes": 1, "No": 0})
df["Alcohol_Use"] = df["Alcohol_Use"].map({"Yes": 1, "No": 0})
df["Risk_Level"] = df["Risk_Level"].map({"Low Risk": 0, "Mid Risk": 1, "HighRisk": 2})
X = df.drop(columns=["Patient_ID", "Risk_Level"])
y = df["Risk_Level"]
X = np.c_[np.ones(X.shape[0]), X]
split_ratios = [0.5, 0.6, 0.7, 0.8]
results = []
for ratio in split_ratios:
  train_size = int(ratio * len(df))
  test_size = len(df) - train_size
  X_train, X_test = X[:train_size], X[train_size:]
  y_train, y_test = y[:train_size], y[train_size:]
  X_train_T = X_train.T
  beta = np.linalg.inv(X_train_T.dot(X_train)).dot(X_train_T).dot(y_train)
  y_pred = X_test.dot(beta)
  y_pred_class = np.round(y_pred)
  y_pred_class = np.clip(y_pred_class, 0, 2)
  accuracy = accuracy_score(y_test, y_pred_class) * 100
  results.append({ "Train-Test Ratio": f"{int(ratio*100)}:{100 - int(ratio*100)}","Accuracy (%)": f"{accuracy:.2f}" })
results_df = pd.DataFrame(results)
print(results_df)
