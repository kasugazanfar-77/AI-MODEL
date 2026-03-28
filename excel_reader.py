"""
ArthSetu AI - Excel Reader Module
----------------------------------
Reads and validates bank statement Excel files.
Expects columns: Date, Description, Amount
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime


def load_bank_statement(filepath: str) -> pd.DataFrame:
    """
    Load and validate a bank statement Excel file.

    Args:
        filepath: Path to the .xlsx or .xls file

    Returns:
        Cleaned and validated DataFrame with Date, Description, Amount columns
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in [".xlsx", ".xls", ".csv"]:
        raise ValueError(f"Unsupported file format: {ext}. Use .xlsx, .xls, or .csv")

    print(f"\n📂 Loading file: {filepath}")

    # Read file based on extension
    try:
        if ext == ".csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")

    print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   📋 Columns found: {list(df.columns)}")

    # ── Column normalization ──────────────────────────────────────────────────
    df.columns = [str(c).strip() for c in df.columns]

    # Try to auto-detect column names (case-insensitive)
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if any(k in lower for k in ["date", "txn date", "transaction date", "value date"]):
            col_map["Date"] = col
        elif any(k in lower for k in ["desc", "narration", "particulars", "remarks", "details"]):
            col_map["Description"] = col
        elif any(k in lower for k in ["amount", "debit", "credit", "withdrawal", "deposit"]):
            col_map["Amount"] = col

    required = ["Date", "Description", "Amount"]
    missing = [r for r in required if r not in col_map]
    if missing:
        # Try exact match as fallback
        for r in required:
            if r in df.columns:
                col_map[r] = r

        still_missing = [r for r in required if r not in col_map]
        if still_missing:
            raise ValueError(
                f"Required columns not found: {still_missing}\n"
                f"File has columns: {list(df.columns)}\n"
                f"Expected: Date, Description, Amount"
            )

    # Rename to standard names
    rename_dict = {v: k for k, v in col_map.items()}
    df = df.rename(columns=rename_dict)

    # Keep only core columns (plus any extras like Balance)
    core_cols = [c for c in ["Date", "Description", "Amount", "Balance", "Type"] if c in df.columns]
    df = df[core_cols].copy()

    # ── Data Cleaning ─────────────────────────────────────────────────────────
    df = _clean_dataframe(df)

    print(f"   ✅ After cleaning: {len(df)} valid transactions")
    return df


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the dataframe."""

    # Drop completely empty rows
    df = df.dropna(how="all")

    # ── Parse Dates ───────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates > 0:
        print(f"   ⚠️  Dropped {invalid_dates} rows with invalid dates")
    df = df.dropna(subset=["Date"])

    # ── Parse Amounts ─────────────────────────────────────────────────────────
    if df["Amount"].dtype == object:
        # Remove currency symbols, commas, spaces
        df["Amount"] = (
            df["Amount"]
            .astype(str)
            .str.replace(r"[₹$€£,\s]", "", regex=True)
            .str.replace(r"\((.+)\)", r"-\1", regex=True)  # (100) → -100
        )
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])

    # ── Clean Descriptions ────────────────────────────────────────────────────
    df["Description"] = df["Description"].astype(str).str.strip()
    df = df[df["Description"].str.len() > 0]
    df = df[df["Description"] != "nan"]

    # ── Add helper columns ────────────────────────────────────────────────────
    df["Month"] = df["Date"].dt.to_period("M")
    df["MonthStr"] = df["Date"].dt.strftime("%b %Y")
    df["DayOfWeek"] = df["Date"].dt.day_name()

    # Consider only debits (expenses) — negative amounts or all amounts
    # We keep all, but tag them
    df["IsExpense"] = df["Amount"] < 0
    df["AbsAmount"] = df["Amount"].abs()

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute high-level summary statistics from the statement.

    Returns:
        dict with total_income, total_expense, net_savings, transaction_count, etc.
    """
    income_df = df[df["Amount"] > 0]
    expense_df = df[df["Amount"] < 0]

    total_income = income_df["Amount"].sum()
    total_expense = expense_df["Amount"].abs().sum()
    net_savings = total_income - total_expense

    return {
        "transaction_count": len(df),
        "income_count": len(income_df),
        "expense_count": len(expense_df),
        "total_income": round(total_income, 2),
        "total_expense": round(total_expense, 2),
        "net_savings": round(net_savings, 2),
        "savings_rate": round((net_savings / total_income * 100) if total_income > 0 else 0, 2),
        "date_range_start": df["Date"].min().strftime("%d %b %Y"),
        "date_range_end": df["Date"].max().strftime("%d %b %Y"),
        "avg_expense_per_txn": round(expense_df["Amount"].abs().mean(), 2) if len(expense_df) > 0 else 0,
    }


def print_summary(df: pd.DataFrame):
    """Print a formatted summary of the loaded statement."""
    stats = get_summary_stats(df)
    print("\n" + "═" * 55)
    print("         📊 BANK STATEMENT SUMMARY")
    print("═" * 55)
    print(f"  📅 Period        : {stats['date_range_start']} → {stats['date_range_end']}")
    print(f"  🔢 Transactions  : {stats['transaction_count']}")
    print(f"  💚 Income txns   : {stats['income_count']}")
    print(f"  🔴 Expense txns  : {stats['expense_count']}")
    print(f"  💰 Total Income  : ₹{stats['total_income']:,.2f}")
    print(f"  💸 Total Expense : ₹{stats['total_expense']:,.2f}")
    print(f"  🏦 Net Savings   : ₹{stats['net_savings']:,.2f}")
    print(f"  📈 Savings Rate  : {stats['savings_rate']:.1f}%")
    print("═" * 55)


if __name__ == "__main__":
    # Quick test
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_bank_statement.xlsx"
    df = load_bank_statement(path)
    print_summary(df)
    print("\nFirst 5 rows:")
    print(df[["Date", "Description", "Amount"]].head())
