import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Create Output Directory
output_dir = r"c:\Users\oneal\VS Code Stuff\MPM-Medical-Algorithm\XGBoost_Real_Output"
os.makedirs(output_dir, exist_ok=True)

print("Loading Real Order Data...")
# Load real backlog data
file_path = r"c:\Users\oneal\VS Code Stuff\MPM-Medical-Algorithm\Manufacturing Data CSVs\Copy of Backlog Report   02 25 2026_mcantu_STRU_BRBacklog_20260128_.csv"
raw_df = pd.read_csv(file_path)

# Data Cleaning and Preparation
# Filter rows where OrderDate and OrderQuantity are valid
df = raw_df.dropna(subset=['OrderDate', 'OrderQuantity']).copy()
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
df = df.dropna(subset=['OrderDate'])

# We will predict the aggregate daily demand for all items (or you could filter by ItemCode)
# Let's group by OrderDate to get total daily units sold and average UnitPrice
daily_df = df.groupby('OrderDate').agg(
    units_sold=('OrderQuantity', 'sum'),
    mpm_average_cost=('UnitPrice', 'mean') # Using unit price as an approximation
).reset_index()

daily_df = daily_df.rename(columns={'OrderDate': 'date'})
daily_df = daily_df.sort_values('date')

# Set date as index to resample and fill missing days with 0
daily_df.set_index('date', inplace=True)
daily_df = daily_df.resample('D').asfreq().fillna({'units_sold': 0})

# Fill forward missing pricing information, or use median
price_median = daily_df['mpm_average_cost'].median()
if pd.isna(price_median):
    price_median = 50.0  # fallback
daily_df['mpm_average_cost'] = daily_df['mpm_average_cost'].fillna(method='ffill').fillna(price_median)

daily_df = daily_df.reset_index()

# Generate dummy competitor pricing since it doesn't exist in the ERP export
np.random.seed(42)
competitor_price = daily_df['mpm_average_cost'] * np.random.uniform(0.9, 1.2, len(daily_df))
daily_df['competitor_price'] = competitor_price

print(f"Loaded {len(daily_df)} days of historical data.")

# Save the prepared real data
output_data_path = os.path.join(output_dir, 'prepared_real_demand.csv')
daily_df.to_csv(output_data_path, index=False)
print(f"Saved prepared real data to {output_data_path}")

# --- ML Pipeline ---
print("Engineering time-series features...")
daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
daily_df['month'] = daily_df['date'].dt.month
daily_df['quarter'] = daily_df['date'].dt.quarter
daily_df['day_of_year'] = daily_df['date'].dt.dayofyear

# Lag Features
daily_df['sales_lag_1'] = daily_df['units_sold'].shift(1)
daily_df['sales_lag_3'] = daily_df['units_sold'].shift(3)
daily_df['sales_lag_7'] = daily_df['units_sold'].shift(7)
daily_df['sales_lag_30'] = daily_df['units_sold'].shift(30)

# Rolling Features
daily_df['rolling_avg_14'] = daily_df['units_sold'].rolling(window=14).mean()
daily_df['rolling_max_14'] = daily_df['units_sold'].rolling(window=14).max()
daily_df['rolling_std_14'] = daily_df['units_sold'].rolling(window=14).std().fillna(0)

# Drop rows with NaN values created by lag/rolling
ml_df = daily_df.dropna().copy()

if len(ml_df) < 50:
    print("WARNING: Not enough historical data after generating 30-day lags to train a good model.")
    # If the real dataset is very small, we might just reduce lag requirements, but we'll try it as-is first.

# Features
features = [
    'competitor_price', 'mpm_average_cost', 'day_of_week', 
    'month', 'quarter', 'day_of_year', 
    'sales_lag_1', 'sales_lag_3', 'sales_lag_7', 'sales_lag_30', 
    'rolling_avg_14', 'rolling_max_14', 'rolling_std_14'
]
X = ml_df[features]
y = ml_df['units_sold']

# Split data (Sequential split for time-series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train
print("Training the XGBoost Regressor on REAL Demand Data...")
model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=300,         
    learning_rate=0.08,        
    max_depth=8,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Eval
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Calculate error margins
mae = mean_absolute_error(y_test, predictions)
print(f"\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f} units")
print(f"(On average, our model's prediction is off by {mae:.2f} units per day)")

with open(os.path.join(output_dir, 'model_evaluation.txt'), 'w') as f:
    f.write(f"Model Evaluation on Real Data:\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.2f} units\n")

# Weightings
print("\nTop 5 Most Important Factors Driving Demand (Why the AI made its prediction):")
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
print(feature_importance_df.head(5).to_string(index=False))

# Plot performance on test set
plt.figure(figsize=(12, 6))
# Only plot maximum last 60 days of test set to keep graph readable
plot_limit = min(60, len(y_test))
plt.plot(ml_df['date'].iloc[-plot_limit:], y_test.values[-plot_limit:], label='Actual Sales', marker='o')
plt.plot(ml_df['date'].iloc[-plot_limit:], predictions[-plot_limit:], label='Predicted Sales', linestyle='dashed', marker='x')

plt.legend()
plt.title('Actual vs Predicted Demand using Real Manufacturing Orders (Test Set)')
plt.ylabel('Units Sold')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()

graph_path = os.path.join(output_dir, 'real_data_actual_vs_predicted.png')
plt.savefig(graph_path)
print(f"\nGraph saved to: {graph_path}")
print("Proof of Concept with Real Data Complete.")
