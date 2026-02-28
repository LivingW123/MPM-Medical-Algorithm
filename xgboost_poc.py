import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Generate Dummy Data
print("Generating 2 years of daily dummy data for 'Component X'...")
np.random.seed(42)

# 730 days
start_date = datetime.now() - timedelta(days=730)
dates = [start_date + timedelta(days=i) for i in range(730)]

# Base demand
base_demand = np.random.normal(50, 10, 730)

# Seasonality
seasonality = np.sin(np.linspace(0, 4 * np.pi, 730)) * 20

# Competitor Pricing
competitor_price = np.random.normal(120, 5, 730)
competitor_impact = (competitor_price - 120) * 1.5

# MPM Buying Price (Cost)
mpm_cost = np.random.normal(80, 5, 730)
demand = np.maximum(base_demand + seasonality + competitor_impact, 0)

df = pd.DataFrame({
    'date': dates,
    'competitor_price': competitor_price,
    'mpm_average_cost': mpm_cost,
    'units_sold': demand
})

# Features
print("Engineering time-series features...")
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['day_of_year'] = df['date'].dt.dayofyear

# Lag Features
df['sales_lag_7'] = df['units_sold'].shift(7)
df['sales_lag_30'] = df['units_sold'].shift(30)

# Rolling Features
df['rolling_avg_14'] = df['units_sold'].rolling(window=14).mean()
df = df.dropna()

# Model Prep
features = [
    'competitor_price', 'mpm_average_cost', 'day_of_week', 
    'month', 'quarter', 'day_of_year', 'sales_lag_7', 
    'sales_lag_30', 'rolling_avg_14'
]
X = df[features]
y = df['units_sold']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train
print("Training the XGBoost Regressor...")
model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,         
    learning_rate=0.1,        
    max_depth=5,              
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

# Weightings
print("\nTop 5 Most Important Factors Driving Demand (Why the AI made its prediction):")
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
print(feature_importance_df.head(5).to_string(index=False))

# 30 days actual vs predicted
plt.plot(y_test.values[-30:], label='Actual Sales')
plt.plot(predictions[-30:], label='Predicted Sales')
plt.legend()
plt.title('Actual vs Predicted Demand: Last 30 Days')
plt.savefig('actual_vs_predicted_demand.png')

print("\nProof of Concept Complete. In production, this model would ingest live API data from your ERP and competitors to output a `recommended_order_quantity`.")
