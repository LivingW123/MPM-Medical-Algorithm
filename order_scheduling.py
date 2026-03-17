import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# ── Configuration ──────────────────────────────────────────────────────────
MFG_START_DATE = pd.Timestamp("2026-02-25")  # backlog report date
BASE = "Manufacturing Data CSVs"
RAW_DATA = os.path.join(BASE, "Copy of Backlog Report   02 25 2026_Raw_data_w_lookup.csv")
LEAD_TIME = os.path.join(BASE, "lead_time.csv")
COST_SHEET = os.path.join(BASE, "Copy of Backlog Report   02 25 2026_Cost_sheet.csv")
OUTPUT_CSV = os.path.join(BASE, "order_schedule_feasibility.csv")
OUTPUT_PNG = os.path.join(BASE, "monthly_revenue_projection.png")

SENTINEL_DATE = pd.Timestamp("1900-01-02")
DEFAULT_LEAD_TIME = 30  # fallback days for items missing lead time data


def load_and_clean_data():
    """Load and normalize the three input CSVs."""
    # ── Raw order data ──
    orders = pd.read_csv(RAW_DATA)
    orders = orders.dropna(subset=["SONumber"])
    orders["ItemCode"] = orders["ItemCode"].astype(str).str.strip()
    for col in ["SONumber", "LineNumber"]:
        orders[col] = orders[col].astype(int)

    date_cols = ["Order date", "Requested Ship date", "ScheduledShipDate", "PromisedShipDate"]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
        orders.loc[orders[col] == SENTINEL_DATE, col] = pd.NaT

    # ── Lead time ──
    lt = pd.read_csv(LEAD_TIME)
    lt = lt.rename(columns={"Item code": "ItemCode", "Lead time (Days)": "lead_time_days"})
    lt["ItemCode"] = lt["ItemCode"].astype(str).str.strip()
    lt = lt[["ItemCode", "lead_time_days"]].drop_duplicates(subset="ItemCode", keep="first")

    # ── Cost sheet ──
    cost = pd.read_csv(COST_SHEET, usecols=["ItemCode", "ItemDescription", "Price", "Cost", "GP", "Gp%"])
    cost["ItemCode"] = cost["ItemCode"].astype(str).str.strip()
    cost = cost.drop_duplicates(subset="ItemCode", keep="first")
    cost = cost.rename(columns={"Gp%": "gp_pct_cost_sheet", "GP": "gp_cost_sheet"})

    return orders, lt, cost


def join_data(orders, lt, cost):
    """Left-join orders with lead time and cost sheet on ItemCode."""
    df = orders.merge(lt, on="ItemCode", how="left")

    missing_lt = df["lead_time_days"].isna().sum()
    if missing_lt:
        missing_items = df.loc[df["lead_time_days"].isna(), "ItemCode"].unique()
        print(f"WARNING: {missing_lt} lines ({len(missing_items)} items) missing lead time, defaulting to {DEFAULT_LEAD_TIME} days")
        print(f"  Items: {', '.join(missing_items[:10])}")
        df["lead_time_days"] = df["lead_time_days"].fillna(DEFAULT_LEAD_TIME)

    df = df.merge(cost[["ItemCode", "gp_pct_cost_sheet"]], on="ItemCode", how="left")

    missing_cost = df["gp_pct_cost_sheet"].isna().sum()
    if missing_cost:
        print(f"INFO: {missing_cost} lines missing cost sheet data, will use raw Profit column")

    print(f"Loaded {len(df)} order lines, {df['SONumber'].nunique()} unique SOs")
    return df


def compute_schedule(df):
    """Compute earliest ship date and feasibility for each order line."""
    df["mfg_start_date"] = MFG_START_DATE
    df["lead_time_days"] = df["lead_time_days"].astype(int)
    df["earliest_ship_date"] = df["mfg_start_date"] + pd.to_timedelta(df["lead_time_days"], unit="D")

    # Feasibility vs customer deadline (Requested Ship date)
    has_deadline = df["Requested Ship date"].notna()
    df["feasibility"] = "NO_DEADLINE"
    df.loc[has_deadline & (df["earliest_ship_date"] <= df["Requested Ship date"]), "feasibility"] = "ON_TIME"
    df.loc[has_deadline & (df["earliest_ship_date"] > df["Requested Ship date"]), "feasibility"] = "LATE"

    # Days margin: positive = ahead of schedule, negative = late
    df["days_margin"] = np.nan
    df.loc[has_deadline, "days_margin"] = (
        (df.loc[has_deadline, "Requested Ship date"] - df.loc[has_deadline, "earliest_ship_date"]).dt.days
    )

    # Profit: use raw Profit column, cross-reference with cost sheet estimate
    df["profit_estimated"] = df["ExtendedLineAmount"] * df["gp_pct_cost_sheet"]

    return df


def compute_revenue(df):
    """Bucket revenue and profit by ship month."""
    df["ship_month"] = df["earliest_ship_date"].dt.to_period("M")

    monthly = df.groupby("ship_month").agg(
        revenue=("ExtendedLineAmount", "sum"),
        profit=("Profit", "sum"),
        profit_estimated=("profit_estimated", "sum"),
        order_count=("SONumber", "count"),
    ).sort_index()

    return monthly


def export_csv(df):
    """Write the line-level schedule to CSV."""
    out_cols = [
        "SONumber", "LineNumber", "ItemCode", "ItemDescription",
        "CustomerCode", "CustomerName", "ItemSalesClassCode",
        "OrderQuantity", "UnitPrice", "ExtendedLineAmount",
        "Order date", "Requested Ship date",
        "lead_time_days", "mfg_start_date", "earliest_ship_date",
        "feasibility", "days_margin",
        "Profit", "gp_pct_cost_sheet", "profit_estimated",
        "ship_month",
    ]
    out = df[[c for c in out_cols if c in df.columns]]
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV saved: {OUTPUT_CSV} ({len(out)} rows)")


def plot_revenue(monthly):
    """Bar chart of monthly revenue and profit."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    months = [str(m) for m in monthly.index]
    x = np.arange(len(months))
    width = 0.6

    # Revenue
    bars1 = ax1.bar(x, monthly["revenue"], width, color="steelblue", label="Revenue")
    ax1.axhline(monthly["revenue"].mean(), color="navy", linestyle="--", alpha=0.5, label="Average")
    ax1.set_ylabel("Revenue ($)")
    ax1.set_title("Projected Monthly Revenue by Earliest Ship Date")
    ax1.legend()
    for bar, val in zip(bars1, monthly["revenue"]):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"\${val:,.0f}", ha="center", va="bottom", fontsize=7)

    # Profit
    bars2 = ax2.bar(x, monthly["profit"], width, color="forestgreen", label="Profit (actual)")
    ax2.axhline(monthly["profit"].mean(), color="darkgreen", linestyle="--", alpha=0.5, label="Average")
    ax2.set_ylabel("Profit ($)")
    ax2.set_title("Projected Monthly Profit by Earliest Ship Date")
    ax2.set_xticks(x)
    ax2.set_xticklabels(months, rotation=45, ha="right")
    ax2.legend()
    for bar, val in zip(bars2, monthly["profit"]):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"\${val:,.0f}", ha="center", va="bottom", fontsize=7)

    total_rev = monthly["revenue"].sum()
    total_prof = monthly["profit"].sum()
    title = f"Total Revenue: \${total_rev:,.0f}  |  Total Profit: \${total_prof:,.0f}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Plot saved: {OUTPUT_PNG}")
    plt.close()


def print_summary(df):
    """Print console summary of scheduling results."""
    print("\n" + "=" * 60)
    print("ORDER SCHEDULING SUMMARY")
    print("=" * 60)
    print(f"Manufacturing start date: {MFG_START_DATE.strftime('%Y-%m-%d')}")
    print(f"Total order lines: {len(df)}")
    print(f"Unique sales orders: {df['SONumber'].nunique()}")

    counts = df["feasibility"].value_counts()
    total = len(df)
    for status in ["ON_TIME", "LATE", "NO_DEADLINE"]:
        n = counts.get(status, 0)
        print(f"  {status}: {n} ({n / total * 100:.1f}%)")

    print(f"\nTotal backlog revenue: ${df['ExtendedLineAmount'].sum():,.2f}")
    print(f"Total backlog profit:  ${df['Profit'].sum():,.2f}")

    late = df[df["feasibility"] == "LATE"].sort_values("ExtendedLineAmount", ascending=False)
    if len(late):
        print(f"\nTop 5 at-risk orders (LATE, by revenue):")
        for _, row in late.head(5).iterrows():
            print(f"  SO {row['SONumber']}-{row['LineNumber']}: "
                  f"{row['ItemDescription'][:45]:45s} "
                  f"${row['ExtendedLineAmount']:>12,.2f}  "
                  f"{int(abs(row['days_margin']))}d late")


def main():
    orders, lt, cost = load_and_clean_data()
    df = join_data(orders, lt, cost)
    df = compute_schedule(df)
    monthly = compute_revenue(df)
    export_csv(df)
    plot_revenue(monthly)
    print_summary(df)


if __name__ == "__main__":
    main()
