"""
Retail Sales Intelligence
IEEE LGU AI/ML Cohort One - Week 3: Data Analysis with NumPy & Pandas
AI/ML Leads: Abdullah Faisal & Alina Irshad

Loads raw sales records from data/sales.csv, cleans them, runs NumPy revenue
analytics, ranks products/categories with groupby, and prints a business report.
Charts are saved as PNG files in the charts/ folder.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed, we only save PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "sales.csv"
CHART_DIR = BASE_DIR / "charts"

LINE = "=" * 62


def banner(title: str) -> None:
    """Print a section heading."""
    print(f"\n{LINE}\n{title}\n{LINE}")


# --------------------------------------------------------------------------
# 1. LOAD & CLEAN
# --------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Read the raw CSV. Everything comes in as text so we can inspect it."""
    return pd.read_csv(path, dtype=str)


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw sales data and return (clean_df, log of what changed)."""
    log = {"rows_loaded": len(raw)}
    df = raw.copy()

    # Tidy text columns: strip spaces, turn empty strings into real NaN
    for col in ["product_name", "category", "region", "date"]:
        df[col] = df[col].str.strip().replace("", np.nan)

    # Convert numeric columns. Anything that is not a number (e.g. "abc") -> NaN
    df["order_id"] = pd.to_numeric(df["order_id"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Duplicate orders: keep the first copy of each order_id
    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    log["duplicates_removed"] = before - len(df)

    # Drop rows we cannot trust or repair:
    #  - no product name or unparseable date (can't say what/when it was)
    #  - units missing, non-numeric, zero or negative (a quantity can't be guessed)
    before = len(df)
    bad_units = df["units_sold"].isna() | (df["units_sold"] <= 0)
    df = df[~bad_units & df["product_name"].notna() & df["date"].notna()]
    log["invalid_rows_dropped"] = before - len(df)

    # Negative prices are data-entry errors -> treat as missing
    negative_price = df["unit_price"] < 0
    log["negative_prices_blanked"] = int(negative_price.sum())
    df.loc[negative_price, "unit_price"] = np.nan

    # Fill missing prices with the median price of the same product.
    # If that product has no valid price at all, fall back to its category median.
    log["prices_filled"] = int(df["unit_price"].isna().sum())
    product_median = df.groupby("product_name")["unit_price"].transform("median")
    category_median = df.groupby("category")["unit_price"].transform("median")
    df["unit_price"] = df["unit_price"].fillna(product_median).fillna(category_median)

    # Region missing -> label as Unknown so the revenue is still counted
    log["regions_filled"] = int(df["region"].isna().sum())
    df["region"] = df["region"].fillna("Unknown")

    df["units_sold"] = df["units_sold"].astype(int)
    df["order_id"] = df["order_id"].astype(int)
    log["rows_clean"] = len(df)
    return df.reset_index(drop=True), log


def print_cleaning_report(log: dict) -> None:
    banner("DATA CLEANING REPORT")
    print(f"Rows loaded from CSV          : {log['rows_loaded']}")
    print(f"Duplicate orders removed      : {log['duplicates_removed']}")
    print(f"Invalid rows dropped          : {log['invalid_rows_dropped']}")
    print(f"Negative prices set to missing: {log['negative_prices_blanked']}")
    print(f"Missing prices filled (median): {log['prices_filled']}")
    print(f"Missing regions -> 'Unknown'  : {log['regions_filled']}")
    print(f"Rows remaining after cleaning : {log['rows_clean']}")


# --------------------------------------------------------------------------
# 2. NUMPY REVENUE METRICS
# --------------------------------------------------------------------------
def add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue = units_sold x unit_price for every row."""
    df = df.copy()
    df["revenue"] = df["units_sold"] * df["unit_price"]
    return df


def business_metrics(df: pd.DataFrame) -> dict:
    """Overall metrics calculated on NumPy arrays."""
    revenue = df["revenue"].to_numpy()
    units = df["units_sold"].to_numpy()
    return {
        "total_revenue": np.sum(revenue),
        "total_orders": len(revenue),
        "aov": np.sum(revenue) / len(revenue),
        "total_units": np.sum(units),
        "median_order": np.median(revenue),
        "max_order": np.max(revenue),
    }


# --------------------------------------------------------------------------
# 3. GROUPBY & FILTERING
# --------------------------------------------------------------------------
def revenue_by(df: pd.DataFrame, column: str) -> pd.Series:
    """Total revenue per group, sorted highest to lowest."""
    return df.groupby(column)["revenue"].sum().sort_values(ascending=False)


def filter_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    return df[df["region"] == region]


def filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)]


def high_value_orders(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return df[df["revenue"] >= threshold].sort_values("revenue", ascending=False)


# --------------------------------------------------------------------------
# 4. CONSOLE REPORT
# --------------------------------------------------------------------------
def print_summary(df: pd.DataFrame, m: dict) -> None:
    banner("RETAIL SALES SUMMARY REPORT")
    print(f"Total Revenue        : ${m['total_revenue']:>10,.2f}")
    print(f"Total Orders         : {m['total_orders']:>11}")
    print(f"Average Order Value  : ${m['aov']:>10,.2f}")
    print(f"Median Order Value   : ${m['median_order']:>10,.2f}")
    print(f"Largest Single Order : ${m['max_order']:>10,.2f}")
    print(f"Total Units Sold     : {m['total_units']:>11}")

    by_product = revenue_by(df, "product_name")
    by_category = revenue_by(df, "category")
    by_region = revenue_by(df, "region")

    banner("PRODUCT RANKING (revenue, high to low)")
    for rank, (name, rev) in enumerate(by_product.items(), start=1):
        print(f"{rank:>2}. {name:<22} ${rev:>9,.2f}")
    print(f"\nHighest-performing product: {by_product.index[0]} (${by_product.iloc[0]:,.2f})")

    banner("CATEGORY RANKING")
    for rank, (name, rev) in enumerate(by_category.items(), start=1):
        share = rev / m["total_revenue"] * 100
        print(f"{rank:>2}. {name:<14} ${rev:>9,.2f}  ({share:4.1f}%)")

    banner("REGION BREAKDOWN")
    for name, rev in by_region.items():
        orders = int((df["region"] == name).sum())
        print(f"{name:<10} ${rev:>9,.2f}   {orders:>2} orders")

    banner("HIGH-VALUE ORDERS (revenue >= $300)")
    top = high_value_orders(df, 300)
    print(top[["order_id", "date", "product_name", "units_sold", "revenue", "region"]]
          .to_string(index=False, formatters={
              "date": lambda d: d.strftime("%Y-%m-%d"),
              "revenue": "${:,.2f}".format,
          }))

    banner("FILTER DEMOS")
    north = filter_region(df, "North")
    print(f"North region      : {len(north)} orders, ${north['revenue'].sum():,.2f}")
    week = filter_dates(df, "2026-09-01", "2026-09-07")
    print(f"Sep 1 - Sep 7     : {len(week)} orders, ${week['revenue'].sum():,.2f}")


# --------------------------------------------------------------------------
# 5. CHARTS (PNG)
# --------------------------------------------------------------------------
def save_charts(df: pd.DataFrame) -> None:
    CHART_DIR.mkdir(exist_ok=True)

    # Revenue by category
    by_category = revenue_by(df, "category")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(by_category.index, by_category.values, color="#2a6fb0")
    ax.set_title("Revenue by Category")
    ax.set_ylabel("Revenue ($)")
    for i, v in enumerate(by_category.values):
        ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "revenue_by_category.png", dpi=150)
    plt.close(fig)

    # Product ranking (horizontal bars, best on top)
    by_product = revenue_by(df, "product_name").sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(by_product.index, by_product.values, color="#2a6fb0")
    ax.set_title("Product Revenue Ranking")
    ax.set_xlabel("Revenue ($)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "product_ranking.png", dpi=150)
    plt.close(fig)

    # Revenue by region
    by_region = revenue_by(df, "region")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(by_region.index, by_region.values, color="#2a6fb0")
    ax.set_title("Revenue by Region")
    ax.set_ylabel("Revenue ($)")
    for i, v in enumerate(by_region.values):
        ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "revenue_by_region.png", dpi=150)
    plt.close(fig)

    # Daily revenue trend
    daily = df.groupby("date")["revenue"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(daily.index, daily.values, marker="o", color="#2a6fb0")
    ax.set_title("Daily Revenue")
    ax.set_ylabel("Revenue ($)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "daily_revenue.png", dpi=150)
    plt.close(fig)

    print(f"\nCharts saved to: {CHART_DIR.name}/ (4 PNG files)")


def main() -> None:
    raw = load_data(DATA_FILE)
    df, log = clean_data(raw)
    print_cleaning_report(log)

    df = add_revenue(df)
    metrics = business_metrics(df)
    print_summary(df, metrics)
    save_charts(df)


if __name__ == "__main__":
    main()
