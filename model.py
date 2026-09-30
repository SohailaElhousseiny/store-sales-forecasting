
import pandas as pd

# Load data
df = pd.read_csv(
    "stores_sales_forecasting.csv",
    encoding="latin1"
)

# Convert date
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Aggregate sales by month
monthly_sales = (
    df.resample("MS", on="Order Date")["Sales"]
    .sum()
    .reset_index()
)

# Create time features
monthly_sales["Year"] = monthly_sales["Order Date"].dt.year
monthly_sales["Month"] = monthly_sales["Order Date"].dt.month
monthly_sales["Quarter"] = monthly_sales["Order Date"].dt.quarter

# Create lag features
monthly_sales["Lag_1"] = monthly_sales["Sales"].shift(1)
monthly_sales["Lag_2"] = monthly_sales["Sales"].shift(2)
monthly_sales["Lag_3"] = monthly_sales["Sales"].shift(3)

# Remove rows with missing lag values
monthly_sales = monthly_sales.dropna()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np

# Define features and target
features = [
    "Year",
    "Month",
    "Quarter",
    "Lag_1",
    "Lag_2",
    "Lag_3"
]

X = monthly_sales[features]
y = monthly_sales["Sales"]

# Split data chronologically
split_index = int(len(monthly_sales) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train the model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Evaluation")
print("------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Save the trained model
joblib.dump(model, "sales_model.pkl")
joblib.dump(features, "features.pkl")

print("\nModel saved successfully!")