# Manufacturing Scheduling Models: Summary

## Data Sources

Both models use the same three input files from `Manufacturing Data CSVs/`:

| Data | Source File | Column(s) Used |
|------|-----------|----------------|
| Sales order identifiers | Raw_data_w_lookup.csv | `SONumber`, `LineNumber` |
| Item & customer info | Raw_data_w_lookup.csv | `ItemCode`, `ItemDescription`, `CustomerCode`, `CustomerName` |
| Order size | Raw_data_w_lookup.csv | `OrderQuantity`, `UnitPrice`, `ExtendedLineAmount` |
| Customer deadline | Raw_data_w_lookup.csv | `Requested Ship date` |
| Gross profit per line | Raw_data_w_lookup.csv | `Profit` |
| Manufacturing lead time | lead_time.csv | `Lead time (Days)` (joined on `Item code` = `ItemCode`) |
| Gross profit % | Cost_sheet.csv | `Gp%` (joined on `ItemCode`, used as cross-reference only) |

**Coverage:** 652 order lines across 478 unique sales orders and 144 unique items. 594 of 652 lines (91.1%) have a customer deadline. 616 of 652 (94.5%) matched the cost sheet. All 652 matched a lead time.

### Assumed (Not From Data)

These parameters are **not in any CSV** and are configurable constants in `batch_scheduling.py`:

| Parameter | Assumed Value | Rationale |
|-----------|--------------|-----------|
| Late penalty rate | 2% of line profit per day late | Industry approximation for customer goodwill / penalty clauses |
| Late penalty cap | 100% of line profit | Cannot lose more than the profit on a line |
| Holding cost | $0.005 / unit / day | Warehousing, insurance, capital cost of finished goods inventory |
| Setup cost | $500 per manufacturing run | Machine changeover, calibration, first-article inspection |
| Manufacturing start date | 2026-02-25 | Backlog report date |

---

## 1. Naive Model (`order_scheduling.py`)

### How It Works

Start manufacturing **everything on 2026-02-25**, regardless of when the customer needs it.

### Equations

```
mfg_start_date  = 2026-02-25                                  (constant for all lines)
earliest_ship   = 2026-02-25 + lead_time_days
days_margin     = requested_ship_date - earliest_ship_date    (positive = early, negative = late)
```

Feasibility flag:
```
if requested_ship_date is missing     -> NO_DEADLINE
if earliest_ship <= requested_ship    -> ON_TIME
if earliest_ship >  requested_ship    -> LATE
```

Cost model:
```
days_late       = max(0, -days_margin)
late_penalty    = min(days_late * 0.02, 1.0) * Profit
days_early      = max(0, days_margin)
holding_cost    = days_early * OrderQuantity * $0.005
setup_cost      = number_of_unique_ItemCodes * $500
adjusted_profit = Profit - late_penalty - holding_cost - setup_cost
```

### Results

| Metric | Value |
|--------|-------|
| On-time | 227 (34.8%) |
| Late | 367 (56.3%) |
| No deadline | 58 (8.9%) |
| Gross profit | $3,849,603 |
| Late penalties | -$680,286 |
| Holding costs | -$1,980,799 |
| Setup costs (144 runs) | -$72,000 |
| **Adjusted profit** | **$1,116,519** |

### Why Holding Costs Are So High

Starting everything on 02-25 means on-time orders ship far ahead of their deadlines. For example:

**SO 64881-1** (URGO VASHE 118ml, qty 244,992, deadline 2026-05-04, lead time 0 days):
```
earliest_ship = 2026-02-25 + 0 days = 2026-02-25
days_early    = 68
holding_cost  = 68 * 244,992 * $0.005 = $83,297
```
The product sits in the warehouse for 68 days waiting for the customer to need it.

**SO 64503-1** (EMED 2.375x2.75 Printed Dress, qty 360,000, deadline 2026-06-30, lead time 14 days):
```
earliest_ship = 2026-02-25 + 14 days = 2026-03-11
days_early    = 111
holding_cost  = 111 * 360,000 * $0.005 = $199,800
```
111 days of warehousing 360,000 units at $0.005/unit/day.

---

## 2. Batched Model (`batch_scheduling.py`)

### How It Works

Group all orders by `ItemCode`. For each group, **backward-schedule from the earliest customer deadline** so the batch ships just in time for the first order that needs it.

```
For each ItemCode:
    earliest_deadline = min(Requested Ship date) across all orders for this item
    ideal_start       = earliest_deadline - lead_time_days
    batch_start       = max(2026-02-25, ideal_start)   # can't start before report date

    All orders for this ItemCode use batch_start as their mfg_start_date
```

### Equations

Same cost model as naive, but `mfg_start_date` varies per item batch:

```
mfg_start_date  = max(2026-02-25, min(deadlines for this ItemCode) - lead_time_days)
earliest_ship   = mfg_start_date + lead_time_days
days_margin     = requested_ship_date - earliest_ship_date
```

All other equations (late_penalty, holding_cost, setup_cost, adjusted_profit) are identical.

### Results

| Metric | Value |
|--------|-------|
| On-time | 227 (34.8%) |
| Late | 367 (56.3%) |
| No deadline | 58 (8.9%) |
| Gross profit | $3,849,603 |
| Late penalties | -$680,286 |
| Holding costs | -$1,419,059 |
| Setup costs (144 runs) | -$72,000 |
| **Adjusted profit** | **$1,678,259** |

81 of 652 order lines received a different (later) start date than naive.
Unique start dates range from 2026-02-25 to 2026-10-01 across 20 distinct dates.

---

## Comparison: Why Batched Is Better

### Side-by-Side

| Metric | Naive | Batched | Delta |
|--------|-------|---------|-------|
| Adjusted profit | $1,116,519 | $1,678,259 | **+$561,741 (+50.3%)** |
| Holding costs | $1,980,799 | $1,419,059 | **-$561,741** |
| Late penalties | $680,286 | $680,286 | $0 |
| Setup costs | $72,000 | $72,000 | $0 |
| On-time count | 227 | 227 | 0 |
| Avg days margin | -70.5 | -77.0 | -6.4 |

### Where the $561,741 Comes From

The improvement is **entirely from reduced holding costs**. Late penalties are identical because 341 of 594 deadlines are already in the past — both models are equally late on those. The batch model saves money on the 227 on-time orders by not manufacturing them too early.

Starting on 02-25 (20 days earlier than 03-17) amplifies the holding cost problem because on-time orders now sit in the warehouse even longer. This is why the batch model's advantage grew from +14.9% (with a 03-17 start) to **+50.3%** (with a 02-25 start) — the earlier the naive model starts, the more inventory waste the batch model eliminates.

### Concrete Examples

**SO 64881-1** (URGO VASHE 118ml, qty 244,992, deadline 2026-05-04, lead time 0 days):
```
NAIVE:   start = 2026-02-25, ship = 2026-02-25, 68 days early, holding = $83,297
BATCHED: start = 2026-05-04, ship = 2026-05-04,  0 days early, holding = $0
Saved:   $83,297
```

**SO 64503-1** (EMED Printed Dress, qty 360,000, deadline 2026-06-30, lead time 14 days):
```
NAIVE:   start = 2026-02-25, ship = 2026-03-11, 111 days early, holding = $199,800
BATCHED: start = 2026-03-17, ship = 2026-03-31,  91 days early, holding = $163,800
Saved:   $36,000
```
This order still ships 91 days early under batching because SO 64501 (same ItemCode, deadline 2026-03-31) pulls the batch start to 2026-03-17. The batch groups all three EMED orders (SOs 64501, 64502, 64503) together, backward-scheduling from the earliest deadline (03-31).

**SO 64501-1** (EMED Printed Dress, qty 360,000, deadline 2026-03-31, lead time 14 days):
```
NAIVE:   start = 2026-02-25, ship = 2026-03-11, 20 days early, holding = $36,000
BATCHED: start = 2026-03-17, ship = 2026-03-31,  0 days early, holding = $0
Saved:   $36,000
```
Backward-scheduled: `2026-03-31 - 14 days = 2026-03-17`. Ships exactly on deadline.

**SO 64966-1** (URGO VASHE 250ml, qty 174,996, deadline 2026-06-01, lead time 0 days):
```
NAIVE:   start = 2026-02-25, ship = 2026-02-25, 96 days early, holding = $83,998
BATCHED: start = 2026-03-23, ship = 2026-03-23, 70 days early, holding = $61,249
Saved:   $22,749
```
Still ships 70 days early because another order for the same item has an earlier deadline (2026-03-23).

### Why On-Time Count Doesn't Change

Both models produce 227 on-time and 367 late orders. This is because:
1. **341 deadlines are already in the past** (before 2026-02-25). Neither model can go back in time, so these are late regardless.
2. For orders with **future deadlines**, backward-scheduling from the deadline guarantees the same on-time result as starting on 02-25 — the batch start is at most 02-25, never later than needed.
3. The batch model **never makes an on-time order late**. It only delays orders that would have shipped unnecessarily early.

### Trade-Off: Average Days Margin

The batch model's average margin is slightly worse (-77.0 vs -70.5), because it intentionally ships on-time orders closer to their deadlines instead of far ahead. This is a feature, not a bug: those 6.4 "lost" days of margin represent the $561,741 in holding cost savings.

### Impact of Start Date Choice

| Start Date | Naive Adj. Profit | Batched Adj. Profit | Batch Advantage |
|-----------|-------------------|---------------------|-----------------|
| 2026-03-17 | $1,331,767 | $1,530,514 | +$198,746 (+14.9%) |
| 2026-02-25 | $1,116,519 | $1,678,259 | +$561,741 (+50.3%) |

The earlier you start naively, the worse the holding cost problem gets. The batch model is robust to start date choice because it independently schedules each item to arrive just in time.
