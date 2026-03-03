import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Volatility-Weighted Kalman Filter (VWKF) & Best Buying Value (BBV) POC
# ==========================================
# This script simulates the price of a medical component (e.g., Laser Diode)
# and applies a VWKF to identify the Best Buying Value (BBV), addressing
# the 3 Profit Leaks:
# 1. Price Leak (Overpaying during market spikes) -> Solved by BBV timing.
# 2. Holding Cost Leak (Buying too much/too early) -> Solved by optimal quantity allocation.
# 3. Stockout Leak (Running out of critical components) -> Solved by safety thresholds.

def simulate_component_price(days=365, initial_price=1500.0):
    """
    Simulates daily price of a medical component (e.g., Laser Diode)
    with changing volatility regimes and mean reversion.
    """
    np.random.seed(42)
    prices = [initial_price]
    volatility = 5.0 # baseline volatility
    
    for i in range(1, days):
        # Introduce volatility shocks
        if 100 < i < 150:
            volatility = 30.0 # High volatility period (e.g. supply chain crisis)
        elif 250 < i < 280:
            volatility = 20.0
        else:
            volatility = 5.0
            
        # Random walk with slight mean reversion to the initial price
        drift = (initial_price - prices[-1]) * 0.02 
        shock = np.random.normal(0, volatility)
        
        new_price = prices[-1] + drift + shock
        prices.append(max(new_price, 100)) # Ensure price stays positive
        
    return np.array(prices)

def apply_vwkf(prices, window=14):
    """
    Volatility-Weighted Kalman Filter (VWKF)
    Adjusts the measurement noise covariance (R) based on recent price volatility.
    High volatility -> higher R -> trust the model more (smooth out the noise).
    Low volatility -> lower R -> trust the measurements more.
    """
    n = len(prices)
    
    # State estimates
    xhat = np.zeros(n)      # A posteriori state estimate
    P = np.zeros(n)         # A posteriori estimate error covariance
    xhatminus = np.zeros(n) # A priori state estimate
    Pminus = np.zeros(n)    # A priori estimate error covariance
    K = np.zeros(n)         # Kalman gain
    
    # Initial guesses
    xhat[0] = prices[0]
    P[0] = 1.0
    
    Q = 1e-3 # Process noise covariance (constant)
    
    # Calculate rolling volatility to weight R (Measurement noise)
    rolling_std = pd.Series(prices).rolling(window=window, min_periods=1).std().fillna(1.0).values
    
    for k in range(1, n):
        # Time update (Predict) - assume random walk model
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        
        # Volatility-Weighted Measurement Noise
        # Base R is scaled by the rolling variance. 
        # When volatility spikes, R increases, meaning we don't overreact to wild price swings.
        R_VW = max(rolling_std[k]**2 * 0.3, 1.0) 
        
        # Measurement update (Correct)
        K[k] = Pminus[k] / (Pminus[k] + R_VW)
        xhat[k] = xhatminus[k] + K[k] * (prices[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return xhat, rolling_std

def calculate_bbv_and_leaks(prices, vwkf_prices, rolling_std, daily_demand=2, initial_inventory=60):
    """
    Identifies Best Buying Value (BBV) signals and simulates inventory 
    to address the 3 Profit Leaks.
    """
    days = len(prices)
    
    # BBV is defined as when the actual price is significantly below the VWKF true value.
    # We use a dynamic threshold based on volatility (e.g., VWKF - 1.5 * StdDev)
    bbv_thresholds = vwkf_prices - (rolling_std * 1.5)
    
    buy_signals = []
    inventory = [initial_inventory]
    
    holding_cost_rate = 0.50 # dollars per unit per day
    stockout_penalty = 1000  # Cost of lost production due to missing part
    
    total_spent = 0
    total_holding_cost = 0
    total_stockout_cost = 0
    
    for k in range(1, days):
        current_inv = inventory[-1] - daily_demand
        
        # Leak 3: Stockout Leak (Must buy regardless of price to avoid huge penalties)
        safety_stock = daily_demand * 7
        if current_inv <= safety_stock:
            order_qty = daily_demand * 30 # Buy a month's worth at market price
            total_spent += order_qty * prices[k]
            buy_signals.append((k, prices[k], 'Urgent/Stockout Avoidance'))
            current_inv += order_qty
            
        # Leak 1 & Leak 2: Price Leak & Holding Leak
        # If price drops below BBV, it's a great time to buy, BUT only if we aren't already overstocked (Leak 2)
        elif prices[k] <= bbv_thresholds[k] and current_inv < daily_demand * 45: 
            order_qty = daily_demand * 60 # Opportunistic bulk buy 
            total_spent += order_qty * prices[k]
            buy_signals.append((k, prices[k], 'BBV Opportunistic'))
            current_inv += order_qty
            
        inventory.append(current_inv)
        
        # Accumulate costs
        if current_inv < 0:
            total_stockout_cost += abs(current_inv) * stockout_penalty
        elif current_inv > 0:
            total_holding_cost += current_inv * holding_cost_rate
            
    # Calculate savings vs naive baseline (buying exactly 30 days worth every 30 days)
    naive_spent = sum(prices[i] * (daily_demand * 30) for i in range(0, days, 30))
    naive_holding = sum((max(0, 30 * daily_demand - (i % 30) * daily_demand) * holding_cost_rate) for i in range(days))
    
    print("\n" + "="*50)
    print("=== 3 PROFIT LEAKS REPORT ===")
    print("="*50)
    
    # Check Price Leak Savings
    price_savings = naive_spent - total_spent
    print(f"1. PRICE LEAK:")
    if price_savings > 0:
        print(f"   [RESOLVED] Saved ${price_savings:,.2f} by tracking BBV instead of fixed interval buying.")
    else:
        print(f"   [WARNING] Cost additional ${-price_savings:,.2f}. Market was highly volatile.")

    # Check Holding Leak Savings
    holding_savings = naive_holding - total_holding_cost
    print(f"\n2. HOLDING LEAK:")
    print(f"   VWKF Holding Cost:  ${total_holding_cost:,.2f}")
    print(f"   Naive Holding Cost: ${naive_holding:,.2f}")
    if holding_savings > 0:
        print(f"   [RESOLVED] Saved ${holding_savings:,.2f} in storage/financing costs.")
    else:
        print(f"   [INFO] Incurred ${-holding_savings:,.2f} extra holding cost to secure BBV deals.")
        
    # Check Stockout Leak
    print(f"\n3. STOCKOUT LEAK:")
    if total_stockout_cost == 0:
        print(f"   [RESOLVED] Zero stockout penalties. Safety thresholds successfully prevented line-down situations.")
    else:
        print(f"   [WARNING] Stockout penalties incurred: ${total_stockout_cost:,.2f}")
        
    print("-" * 50)
    print(f"Final VWKF Total Cost:  ${(total_spent + total_holding_cost + total_stockout_cost):,.2f}")
    print(f"Final Naive Total Cost: ${(naive_spent + naive_holding):,.2f}")
    print(f"TOTAL SYSTEM SAVINGS:   ${(naive_spent + naive_holding) - (total_spent + total_holding_cost + total_stockout_cost):,.2f}")
    print("=" * 50 + "\n")

    return bbv_thresholds, buy_signals, inventory

if __name__ == "__main__":
    print("-> Simulating prices for 'Laser Diode Component' over 1 year...")
    prices = simulate_component_price(days=365, initial_price=1200.0)
    
    print("-> Applying Volatility-Weighted Kalman Filter (VWKF)...")
    vwkf_prices, rolling_std = apply_vwkf(prices, window=14)
    
    print("-> Calculating Best Buying Value (BBV) signals...")
    bbv_thresholds, buy_signals, inventory = calculate_bbv_and_leaks(prices, vwkf_prices, rolling_std)
    
    # Generate visualization
    print("-> Generating analysis visualization (vwkf_bbv_analysis.png)...")
    plt.figure(figsize=(14, 6))
    
    # Plot - Price and BBV
    plt.plot(prices, label='Actual Component Price', color='gray', alpha=0.5, linewidth=1)
    plt.plot(vwkf_prices, label='VWKF Smoothed Price', color='blue', linewidth=2)
    plt.plot(bbv_thresholds, label='BBV Action Threshold', color='green', linestyle='dashed', alpha=0.8)
    
    # Annotate buy signals
    bbv_buys_x = [b[0] for b in buy_signals if 'BBV' in b[2]]
    bbv_buys_y = [b[1] for b in buy_signals if 'BBV' in b[2]]
    plt.scatter(bbv_buys_x, bbv_buys_y, color='green', marker='^', s=120, label='BBV Opportunistic Buy', zorder=5)
    
    urgent_buys_x = [b[0] for b in buy_signals if 'Urgent' in b[2]]
    urgent_buys_y = [b[1] for b in buy_signals if 'Urgent' in b[2]]
    plt.scatter(urgent_buys_x, urgent_buys_y, color='red', marker='o', s=100, label='Urgent Safety Buy', zorder=5)
        
    plt.title('VWKF Price Smoothing & Best Buying Value (BBV) Signals', fontsize=14)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vwkf_bbv_analysis.png', dpi=300)
    print("-> Script execution completed. Output saved to 'vwkf_bbv_analysis.png'")
