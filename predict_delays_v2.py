

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1) TRAIN ON 2024
# -----------------------------
delay_2024 = pd.read_csv("master sheet 24.csv")
delay_2024["period"] = pd.to_datetime(
    delay_2024["reporting_period"].astype(str) + "01",
    format="%Y%m%d", errors="coerce"
)
delay_2024["delay_mins"] = pd.to_numeric(
    delay_2024["average_delay_mins"], errors="coerce"
)

monthly_delay_2024 = (
    delay_2024.dropna(subset=["period", "delay_mins"])
              .groupby("period", as_index=False)["delay_mins"]
              .mean()
)

weather_df = pd.read_csv("Weather Data Top 10 UK Airports 2010-2025 (processed) - Weather data.csv")
weather_df["time"] = pd.to_datetime(weather_df["time"], errors="coerce")

weather_2024 = weather_df[weather_df["time"].dt.year == 2024].copy()
weather_2024["period"] = weather_2024["time"].dt.to_period("M").dt.to_timestamp()

DEW = "dew_point_2m_max (°C)"
PRECIP = "precipitation_hours (h)"
WIND_DIR = "wind_direction_10m_dominant (°)"

monthly_weather_2024 = (
    weather_2024.groupby("period", as_index=False)[[DEW, PRECIP, WIND_DIR]]
                .mean()
)

merged_2024 = pd.merge(monthly_delay_2024, monthly_weather_2024,
                       on="period", how="inner").dropna()

# Wind direction cyclic encoding
theta_2024 = np.deg2rad(merged_2024[WIND_DIR])
merged_2024["sin"] = np.sin(theta_2024)
merged_2024["cos"] = np.cos(theta_2024)

X_train = merged_2024[[DEW, PRECIP, "sin", "cos"]]
y_train = merged_2024["delay_mins"]

model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model trained on 2024 Heathrow data.")

# -----------------------------
# 2) PREDICT & VALIDATE 2025
# -----------------------------
delay_2025 = pd.read_csv("Copy of Master Punctuality sheet (2025) - processed - Sheet1.csv")

delay_2025["period"] = pd.to_datetime(
    delay_2025["reporting_period"].astype(str) + "01",
    format="%Y%m%d", errors="coerce"
)
delay_2025["delay_mins"] = pd.to_numeric(delay_2025["average_delay_mins"], errors="coerce")

monthly_delay_2025 = (
    delay_2025.dropna(subset=["period", "delay_mins"])
              .groupby("period", as_index=False)["delay_mins"]
              .mean()
              .sort_values("period")
)

weather_2025 = weather_df[weather_df["time"].dt.year == 2025].copy()
weather_2025["period"] = weather_2025["time"].dt.to_period("M").dt.to_timestamp()

monthly_weather_2025 = (
    weather_2025.groupby("period", as_index=False)[[DEW, PRECIP, WIND_DIR]]
                .mean()
                .sort_values("period")
)

merged_2025 = pd.merge(monthly_delay_2025, monthly_weather_2025, on="period", how="inner").dropna()

theta_2025 = np.deg2rad(merged_2025[WIND_DIR])
merged_2025["sin"] = np.sin(theta_2025)
merged_2025["cos"] = np.cos(theta_2025)

X_test = merged_2025[[DEW, PRECIP, "sin", "cos"]]
y_test = merged_2025["delay_mins"]

pred_2025 = model.predict(X_test)

# Metrics
print("\n📊 2025 Holdout Performance:")
print(f"R²: {r2_score(y_test, pred_2025):.2f}")
print(f"MAE: {mean_absolute_error(y_test, pred_2025):.1f} mins")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_2025)):.1f} mins")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(merged_2025["period"], y_test, "o-", label="Actual 2025 Delays", linewidth=2)
plt.plot(merged_2025["period"], pred_2025, "s-", label="Predicted (from 2024 Model)", linewidth=2)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
plt.title("Heathrow: 2025 Actual vs Predicted Delays\n(Linear Regression: Weather → Delays, Trained on 2024)", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Average Delay (minutes)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("2025_validation.png", dpi=300, bbox_inches='tight')  # For report
plt.show()

print("\n✅ Plot saved as '2025_validation.png'")
