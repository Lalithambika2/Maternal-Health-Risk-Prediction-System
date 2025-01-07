import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
df = pd.read_csv("maternal_health_dataset.csv")
df["Smoking_Status"] = df["Smoking_Status"].map({"Yes": 1, "No": 0})
df["Alcohol_Use"] = df["Alcohol_Use"].map({"Yes": 1, "No": 0})
df["Risk_Level"] = df["Risk_Level"].map({"Low Risk": 0, "Mid Risk": 1, "HighRisk": 2})
X = df.drop(columns=["Patient_ID", "Risk_Level"]) # Drop Patient_ID as it's not a feature
y = df["Risk_Level"]
split_ratios = [0.5, 0.6, 0.7, 0.8]
results = []
for ratio in split_ratios:
  train_size = int(ratio * len(df))
  X_train, X_test = X[:train_size], X[train_size:]
  y_train, y_test = y[:train_size], y[train_size:]
  model = GaussianNB()
  model.fit(X_train, y_train)
  # Make predictions and calculate accuracy
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred) * 100
  train_ratio = int(ratio * 100)
  test_ratio = 100 - train_ratio
  results.append({"Train-Test Ratio": f"{train_ratio}:{test_ratio}", "Accuracy(%)": f"{accuracy:.2f}"})
results_df = pd.DataFrame(results)
print(results_df)
