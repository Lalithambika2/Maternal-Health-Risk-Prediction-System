import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
df = pd.read_csv("maternal_health_dataset.csv")
df["Smoking_Status"] = df["Smoking_Status"].map({"Yes": 1, "No": 0})
df["Alcohol_Use"] = df["Alcohol_Use"].map({"Yes": 1, "No": 0})
df["Risk_Level"] = df["Risk_Level"].map({"Low Risk": 0, "Mid Risk": 1, "HighRisk": 2})
X = df.drop(columns=["Patient_ID", "Risk_Level"])
y = df["Risk_Level"]
split_ratios = [0.5, 0.6, 0.7, 0.8]
results = []
for ratio in split_ratios:
  train_size = int(ratio * len(df))
  test_size = len(df) - train_size
  X_train, X_test, y_train, y_test = train_test_split(
  X, y, train_size=train_size, test_size=test_size, shuffle=True,random_state=42)
  model = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='ovr')
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred) * 100
  results.append({ "Train-Test Ratio": f"{int(ratio*100)}:{100 - int(ratio*100)}","Accuracy (%)": f"{accuracy:.2f}”})
results_df = pd.DataFrame(results)
print(results_df)
