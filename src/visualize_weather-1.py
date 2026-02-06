import pandas as pd
import numpy as np

# Load CSV exported from Google Sheets
df = pd.read_csv("Main (2022-2024) - Main.csv")

# Parse date
df["reporting_period"] = pd.to_datetime(df["reporting_period"])
df["year"] = df["reporting_period"].dt.year
df["month"] = df["reporting_period"].dt.month

# Target variable
y = df["Median Delay (in decimals)"]

weather_features = [
    "Hot Day (>30)",
    "Cold Day (<0)",
    "Heavy Rain Day (>10)",
    "Heavy Snow Day (>1)",
    "Persistent Rain (>=6)",
    "Overcast (>80)",
    "Fog Risk (>90)",
    "High Wind Day (>15)",
    "Strong Sustained Wind (>12)"
]

weather_cols = [
    "Hot Day (>30)",
    "Cold Day (<0)",
    "Heavy Rain Day (>10)",
    "Heavy Snow Day (>1)",
    "Persistent Rain (>=6)",
    "Overcast (>80)",
    "Fog Risk (>90)",
    "High Wind Day (>15)",
    "Strong Sustained Wind (>12)"
]

df["Weather_Stress_Index"] = df[weather_cols].sum(axis=1)

categorical_features = [
    "reporting_airport",
    "airline_name",
    "month"
]

X = df[['Weather_Stress_Index'] + categorical_features]

train_mask = df["year"] <= 2023
test_mask = df["year"] == 2024

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", ['Weather_Stress_Index'])
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ]
)

model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

y_pred = model.predict(X_test)

# Optional: clip to valid range
y_pred = np.clip(y_pred, 0, 1)

rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.3f}")

feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["regressor"].coef_

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

coef_df.head(10)

# Predict on test period
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 1)

# Build results dataframe
results = df.loc[test_mask, [
    "reporting_period",
    "reporting_airport",
    "airline_name"
]].copy()

results["Actual Delay"] = y_test.values
results["Predicted Delay"] = y_pred

results["month"] = results["reporting_period"].dt.to_period("M").dt.to_timestamp()

time_series = (
    results
    .groupby("month")[["Actual Delay", "Predicted Delay"]]
    .mean()
    .reset_index()
)

full_2024 = pd.date_range(
    start="2024-01-01",
    end="2024-12-01",
    freq="MS"
)

time_series = (
    time_series
    .set_index("month")
    .reindex(full_2024)
    .reset_index()
    .rename(columns={"index": "month"})
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(
    time_series["month"],
    time_series["Actual Delay"]
)

plt.xticks(
    full_2024,
    full_2024.strftime("%Y-%m"),
    rotation=45
)

plt.xlabel("Month")
plt.ylabel("Average delay percentage")
plt.title("Actual average monthly flight delay percentage (test period)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(
    time_series["month"],
    time_series["Predicted Delay"]
)

plt.xticks(
    full_2024,
    full_2024.strftime("%Y-%m"),
    rotation=45
)

plt.xlabel("Month")
plt.ylabel("Average delay percentage")
plt.title("Predicted average monthly flight delay percentage (test period)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(
    time_series["month"],
    time_series["Actual Delay"],
    label="Actual"
)

plt.xticks(
    full_2024,
    full_2024.strftime("%Y-%m"),
    rotation=45
)

plt.plot(
    time_series["month"],
    time_series["Predicted Delay"],
    linestyle="--",
    label="Predicted"
)

plt.xticks(
    full_2024,
    full_2024.strftime("%Y-%m"),
    rotation=45
)

plt.xlabel("Month")
plt.ylabel("Average delay percentage")
plt.title("Actual vs predicted monthly flight delay percentages")
plt.legend()
plt.tight_layout()
plt.show()
