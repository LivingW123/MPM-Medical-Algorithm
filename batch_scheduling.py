"""
Batch Scheduling Optimizer
--------------------------
Instead of starting all manufacturing today (naive approach from order_scheduling.py),
this script groups orders by ItemCode into batches and finds the optimal start date
for each batch to maximize profit after accounting for:

  - Late penalty:   orders that miss the customer deadline lose a % of profit per day late
  - Holding cost:   orders that ship early sit in inventory, costing $/unit/day
  - Batch savings:  running one manufacturing batch per item instead of many reduces setup costs

The optimizer backward-schedules from customer deadlines, then compares total adjusted
profit against the naive "start everything today" baseline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ── Reuse data loading from order_scheduling ──────────────────────────────
from order_scheduling import load_and_clean_data, join_data

# ── Configuration ─────────────────────────────────────────────────────────
TODAY = pd.Timestamp("2026-02-25")  # backlog report date
BASE = "Manufacturing Data CSVs"
OUTPUT_CSV = os.path.join(BASE, "batch_schedule_feasibility.csv")
OUTPUT_PNG = os.path.join(BASE, "batch_vs_naive_comparison.png")

# Cost model parameters (tunable)
LATE_PENALTY_PER_DAY = 0.02       # 2% of line profit lost per day late
LATE_PENALTY_CAP = 1.0            # max 100% of profit lost
HOLDING_COST_PER_UNIT_PER_DAY = 0.005  # $0.005/unit/day sitting in inventory
SETUP_COST_PER_RUN = 500.0       # fixed cost per manufacturing batch start


# ── Naive scheduler (mirrors order_scheduling.py) ────────────────────────

def schedule_naive(df):
    """Start everything today. Returns df with adjusted profit."""
    df = df.copy()
    df["mfg_start_date"] = TODAY
    df["earliest_ship_date"] = TODAY + pd.to_timedelta(df["lead_time_days"], unit="D")
    df = _compute_costs(df)
    df["strategy"] = "naive"
    return df


# ── Batch scheduler ──────────────────────────────────────────────────────

def schedule_batched(df):
    """
    For each ItemCode, find the optimal single start date that maximizes
    adjusted profit across all orders for that item.

    Strategy per batch:
      1. Backward-schedule from the EARLIEST deadline in the batch:
         ideal_start = min(deadlines) - lead_time
      2. Clamp to today (can't start in the past)
      3. All orders in the batch share this start date and ship together
      4. One setup cost per batch instead of per-order
    """
    df = df.copy()
    results = []

    for item_code, group in df.groupby("ItemCode"):
        lead_time = group["lead_time_days"].iloc[0]
        has_deadline = group["Requested Ship date"].notna()

        if has_deadline.any():
            # Backward-schedule from the earliest customer deadline
            earliest_deadline = group.loc[has_deadline, "Requested Ship date"].min()
            ideal_start = earliest_deadline - pd.Timedelta(days=lead_time)
            # Can't start before today
            batch_start = max(TODAY, ideal_start)
        else:
            # No deadlines at all — no rush, start today
            batch_start = TODAY

        group["mfg_start_date"] = batch_start
        group["earliest_ship_date"] = batch_start + pd.Timedelta(days=lead_time)
        results.append(group)

    df = pd.concat(results, ignore_index=True)
    df = _compute_costs(df)
    df["strategy"] = "batched"
    return df


# ── Cost model ───────────────────────────────────────────────────────────

def _compute_costs(df):
    """Apply late penalties and holding costs to compute adjusted profit."""
    has_deadline = df["Requested Ship date"].notna()

    # Days margin: positive = early, negative = late
    df["days_margin"] = np.nan
    df.loc[has_deadline, "days_margin"] = (
        (df.loc[has_deadline, "Requested Ship date"] - df.loc[has_deadline, "earliest_ship_date"]).dt.days
    )

    # Feasibility flag
    df["feasibility"] = "NO_DEADLINE"
    df.loc[has_deadline & (df["days_margin"] >= 0), "feasibility"] = "ON_TIME"
    df.loc[has_deadline & (df["days_margin"] < 0), "feasibility"] = "LATE"

    # Late penalty: % of profit lost per day late, capped
    df["days_late"] = np.clip(-df["days_margin"].fillna(0), 0, None)
    df["late_penalty_pct"] = np.clip(df["days_late"] * LATE_PENALTY_PER_DAY, 0, LATE_PENALTY_CAP)
    df["late_penalty"] = df["Profit"].fillna(0) * df["late_penalty_pct"]

    # Holding cost: early shipments sit in inventory
    df["days_early"] = np.clip(df["days_margin"].fillna(0), 0, None)
    df["holding_cost"] = df["days_early"] * df["OrderQuantity"] * HOLDING_COST_PER_UNIT_PER_DAY

    # Adjusted profit per line
    df["adjusted_profit"] = df["Profit"].fillna(0) - df["late_penalty"] - df["holding_cost"]

    return df


def compute_setup_costs(df):
    """Count distinct manufacturing runs and total setup cost."""
    # Each unique (ItemCode, mfg_start_date) pair = one manufacturing run
    runs = df.groupby(["ItemCode", "mfg_start_date"]).ngroups
    return runs, runs * SETUP_COST_PER_RUN


# ── Comparison ───────────────────────────────────────────────────────────

def compare(naive_df, batch_df):
    """Print side-by-side comparison of both strategies."""
    print("\n" + "=" * 72)
    print("BATCH vs NAIVE SCHEDULING COMPARISON")
    print("=" * 72)

    naive_runs, naive_setup = compute_setup_costs(naive_df)
    batch_runs, batch_setup = compute_setup_costs(batch_df)

    metrics = {}
    for label, sdf in [("NAIVE", naive_df), ("BATCHED", batch_df)]:
        runs, setup = compute_setup_costs(sdf)
        counts = sdf["feasibility"].value_counts()
        total = len(sdf)

        m = {
            "total_lines": total,
            "on_time": counts.get("ON_TIME", 0),
            "late": counts.get("LATE", 0),
            "no_deadline": counts.get("NO_DEADLINE", 0),
            "on_time_pct": counts.get("ON_TIME", 0) / total * 100,
            "late_pct": counts.get("LATE", 0) / total * 100,
            "gross_profit": sdf["Profit"].fillna(0).sum(),
            "late_penalty": sdf["late_penalty"].sum(),
            "holding_cost": sdf["holding_cost"].sum(),
            "setup_cost": setup,
            "mfg_runs": runs,
            "adjusted_profit": sdf["adjusted_profit"].sum() - setup,
            "revenue": sdf["ExtendedLineAmount"].sum(),
            "avg_days_margin": sdf.loc[sdf["feasibility"] != "NO_DEADLINE", "days_margin"].mean(),
        }
        metrics[label] = m

    # Print table
    header = f"{'Metric':<30s} {'NAIVE':>18s} {'BATCHED':>18s} {'Delta':>14s}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Order lines", "total_lines", "{:,.0f}", False),
        ("On-time", "on_time", "{:,.0f}", True),
        ("On-time %", "on_time_pct", "{:.1f}%", True),
        ("Late", "late", "{:,.0f}", False),
        ("Late %", "late_pct", "{:.1f}%", False),
        ("Mfg runs (batches)", "mfg_runs", "{:,.0f}", False),
        ("", None, None, None),
        ("Gross profit", "gross_profit", "${:,.0f}", True),
        ("Late penalties", "late_penalty", "${:,.0f}", False),
        ("Holding costs", "holding_cost", "${:,.0f}", False),
        ("Setup costs", "setup_cost", "${:,.0f}", False),
        ("", None, None, None),
        ("ADJUSTED PROFIT", "adjusted_profit", "${:,.0f}", True),
        ("Avg days margin", "avg_days_margin", "{:.1f}", True),
    ]

    for label, key, fmt, higher_better in rows:
        if key is None:
            print()
            continue
        nv = metrics["NAIVE"][key]
        bv = metrics["BATCHED"][key]
        delta = bv - nv
        delta_str = fmt.format(delta)
        if delta > 0:
            delta_str = "+" + delta_str
        print(f"  {label:<28s} {fmt.format(nv):>18s} {fmt.format(bv):>18s} {delta_str:>14s}")

    improvement = metrics["BATCHED"]["adjusted_profit"] - metrics["NAIVE"]["adjusted_profit"]
    pct = improvement / abs(metrics["NAIVE"]["adjusted_profit"]) * 100 if metrics["NAIVE"]["adjusted_profit"] != 0 else 0
    print(f"\n  Batch scheduling {'improves' if improvement > 0 else 'reduces'} adjusted profit by \${abs(improvement):,.0f} ({abs(pct):.1f}%)")

    return metrics


def plot_comparison(naive_df, batch_df, metrics):
    """Side-by-side bar charts comparing both strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Feasibility breakdown
    ax = axes[0, 0]
    labels = ["ON_TIME", "LATE", "NO_DEADLINE"]
    naive_counts = [metrics["NAIVE"][k.lower()] for k in labels]
    batch_counts = [metrics["BATCHED"][k.lower()] for k in labels]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, naive_counts, w, color="salmon", label="Naive")
    ax.bar(x + w/2, batch_counts, w, color="steelblue", label="Batched")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Order lines")
    ax.set_title("Feasibility: Naive vs Batched")
    ax.legend()

    # 2. Profit waterfall
    ax = axes[0, 1]
    categories = ["Gross\nProfit", "Late\nPenalty", "Holding\nCost", "Setup\nCost", "Adjusted\nProfit"]
    for i, (label, color) in enumerate([("Naive", "salmon"), ("Batched", "steelblue")]):
        m = metrics[label.upper()]
        vals = [m["gross_profit"], -m["late_penalty"], -m["holding_cost"], -m["setup_cost"], m["adjusted_profit"]]
        offset = -0.18 + i * 0.36
        bars = ax.bar(np.arange(len(categories)) + offset, vals, 0.35, color=color, label=label)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Dollars")
    ax.set_title("Profit Breakdown")
    ax.legend()

    # 3. Monthly revenue comparison
    ax = axes[1, 0]
    for label, sdf, color in [("Naive", naive_df, "salmon"), ("Batched", batch_df, "steelblue")]:
        sdf = sdf.copy()
        sdf["ship_month"] = sdf["earliest_ship_date"].dt.to_period("M")
        monthly = sdf.groupby("ship_month")["ExtendedLineAmount"].sum().sort_index()
        months = [str(m) for m in monthly.index]
        ax.bar(np.arange(len(months)) + (-0.18 if label == "Naive" else 0.18),
               monthly.values, 0.35, color=color, label=label)
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Monthly Revenue Distribution")
    ax.legend()
    # Use the wider set of months for x labels
    naive_months = naive_df.copy()
    naive_months["ship_month"] = naive_months["earliest_ship_date"].dt.to_period("M")
    batch_months = batch_df.copy()
    batch_months["ship_month"] = batch_months["earliest_ship_date"].dt.to_period("M")
    all_months = sorted(set(naive_months["ship_month"].unique()) | set(batch_months["ship_month"].unique()))
    ax.set_xticks(np.arange(len(all_months)))
    ax.set_xticklabels([str(m) for m in all_months], rotation=45, ha="right", fontsize=8)

    # 4. Days margin histogram
    ax = axes[1, 1]
    for label, sdf, color in [("Naive", naive_df, "salmon"), ("Batched", batch_df, "steelblue")]:
        margins = sdf.loc[sdf["feasibility"] != "NO_DEADLINE", "days_margin"].dropna()
        ax.hist(margins, bins=30, alpha=0.6, color=color, label=label, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="Deadline")
    ax.set_xlabel("Days margin (negative = late)")
    ax.set_ylabel("Order lines")
    ax.set_title("Schedule Margin Distribution")
    ax.legend()

    fig.suptitle("Batch Scheduling vs Naive Scheduling", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Plot saved: {OUTPUT_PNG}")
    plt.close()


def export_batch_csv(batch_df):
    """Write the batch-scheduled results to CSV."""
    out_cols = [
        "SONumber", "LineNumber", "ItemCode", "ItemDescription",
        "CustomerCode", "CustomerName",
        "OrderQuantity", "UnitPrice", "ExtendedLineAmount",
        "Order date", "Requested Ship date",
        "lead_time_days", "mfg_start_date", "earliest_ship_date",
        "feasibility", "days_margin",
        "Profit", "late_penalty", "holding_cost", "adjusted_profit",
        "ship_month",
    ]
    batch_df = batch_df.copy()
    batch_df["ship_month"] = batch_df["earliest_ship_date"].dt.to_period("M")
    out = batch_df[[c for c in out_cols if c in batch_df.columns]]
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV saved: {OUTPUT_CSV} ({len(out)} rows)")


def main():
    # Load data (reuses order_scheduling's cleaning + joins)
    orders, lt, cost = load_and_clean_data()
    df = join_data(orders, lt, cost)
    df["lead_time_days"] = df["lead_time_days"].astype(int)

    # Run both strategies
    naive_df = schedule_naive(df)
    batch_df = schedule_batched(df)

    # Compare
    metrics = compare(naive_df, batch_df)
    plot_comparison(naive_df, batch_df, metrics)
    export_batch_csv(batch_df)


if __name__ == "__main__":
    main()
