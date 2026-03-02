import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# Optimal Purchasing Algorithm POC
# ==========================================
# This script simulates the decision logic of when to buy medical components.
# It uses inputs that would typically come from:
# 1. xgboost_poc.py (Demand Forecasting)
# 2. ai_scraper_poc.py (Live Prices & Volume Discounts)
# 
# Key features:
# - Calculates total cost (Purchase Price + Holding/Storage Cost)
# - Evaluates Volume Discounts
# - Synchronizes deliveries: penalizes orders that arrive at different times 
#   to save warehouse space and handling costs.

def calculate_optimal_purchase(components_data, days_to_look_ahead=30, holding_cost_per_unit_per_day=0.05, sync_delivery_bonus=50):
    """
    Evaluates purchasing strategies for multiple components to find the optimal order timing.
    """
    print(f"--- Running Optimal Purchasing Algorithm ({days_to_look_ahead} Day Window) ---")
    
    # Simple simulation: compare "Buy Now" vs "Buy Later (sync)" 
    # For a real implementation, this would be a linear programming or dynamic programming solver.
    
    results = {}
    
    for comp_name, data in components_data.items():
        print(f"\nAnalyzing {comp_name}:")
        current_inventory = data['current_inventory']
        daily_demand = data['predicted_daily_demand'] # From XGBoost
        lead_time_days = data['lead_time_days']
        
        # Determine when we will run out of stock
        days_until_stockout = current_inventory / daily_demand
        order_deadline = max(0, int(days_until_stockout - lead_time_days))
        
        print(f"  Current Inventory: {current_inventory}, Daily Demand: {daily_demand}")
        print(f"  Days until stockout: {days_until_stockout:.1f} days. Must order within {order_deadline} days.")
        
        # Evaluate pricing tiers (from AI Scraper)
        required_quantity = int(daily_demand * days_to_look_ahead)
        best_price_per_unit = data['base_price']
        
        # Check if a volume discount applies or if we should buy more to hit a tier
        for tier in data['volume_discounts']:
            if required_quantity >= tier['min_quantity']:
                best_price_per_unit = tier['price_per_unit']
        
        print(f"  Target Quantity for {days_to_look_ahead} days: {required_quantity} units.")
        print(f"  Applicable Price: ${best_price_per_unit:.2f}/unit")
        
        cost_of_goods = required_quantity * best_price_per_unit
        
        # Calculate holding cost if we buy today vs waiting until the deadline
        # If we buy today, it arrives in `lead_time_days`. We hold it for `order_deadline` extra days.
        extra_holding_days = order_deadline
        holding_cost = required_quantity * extra_holding_days * holding_cost_per_unit_per_day
        
        results[comp_name] = {
            'order_quantity': required_quantity,
            'must_order_by_days': order_deadline,
            'lead_time_days': lead_time_days,
            'cost_of_goods': cost_of_goods,
            'holding_cost_if_bought_now': holding_cost
        }
        
    # --- Synchronization Logic ---
    # To save storage space and handling, we want items to arrive on the same day.
    print("\n--- Delivery Synchronization Analysis ---")
    
    # Find common arrival days if we ordered today
    arrival_days = {comp: data['lead_time_days'] for comp, data in components_data.items()}
    max_lead_time = max(arrival_days.values())
    
    print(f"If we order everything today, items arrive on days: {arrival_days}")
    print("To synchronize deliveries to save warehouse staging space:")
    
    for comp_name, data in results.items():
        # How much to delay the order so it arrives with the slowest item
        delay_days = max_lead_time - data['lead_time_days']
        
        if delay_days > data['must_order_by_days']:
            print(f"  [WARNING] Cannot sync {comp_name} to arrive on day {max_lead_time}, it will stock out first!")
            recommended_action = f"Order {data['order_quantity']} ASAP"
        elif delay_days > 0:
            print(f"  [SYNC] Delay {comp_name} order by {delay_days} days to arrive with other components.")
            recommended_action = f"Order {data['order_quantity']} in {delay_days} days"
        else:
            print(f"  [SYNC] Order {comp_name} today (Longest lead time).")
            recommended_action = f"Order {data['order_quantity']} today"
            
        results[comp_name]['recommended_action'] = recommended_action
        
    print("\n--- Final Purchasing Recommendations ---")
    for comp, data in results.items():
        print(f"{comp}: {data['recommended_action']} (Cost: ${data['cost_of_goods']:.2f})")

if __name__ == "__main__":
    # Mock data synthesizing outputs from XGBoost & AI Scraper
    mock_supply_chain_data = {
        "10mL Syringes": {
            "predicted_daily_demand": 120,    # From XGBoost
            "current_inventory": 1500,
            "lead_time_days": 5,
            "base_price": 0.25,               # From Scraper
            "volume_discounts": [             # From Scraper
                {"min_quantity": 1000, "price_per_unit": 0.20},
                {"min_quantity": 5000, "price_per_unit": 0.15}
            ]
        },
        "ECG Electrodes": {
            "predicted_daily_demand": 300,    # From XGBoost
            "current_inventory": 4000,
            "lead_time_days": 8,
            "base_price": 0.15,               # From Scraper
            "volume_discounts": [             # From Scraper
                {"min_quantity": 5000, "price_per_unit": 0.10},
                {"min_quantity": 10000, "price_per_unit": 0.08}
            ]
        },
        "Surgical Masks": {
            "predicted_daily_demand": 500,
            "current_inventory": 8000,
            "lead_time_days": 2,
            "base_price": 0.05,
            "volume_discounts": [
                {"min_quantity": 10000, "price_per_unit": 0.04}
            ]
        }
    }
    
    calculate_optimal_purchase(mock_supply_chain_data)
