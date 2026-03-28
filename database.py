"""
ArthSetu AI - Local SQLite Database Module
-------------------------------------------
Stores all transaction data, risk assessments, and analysis results
in a fully local SQLite database. No cloud, no internet required.

Schema:
  - transactions     : Individual classified transactions
  - category_summary : Aggregated category spending
  - risk_assessments : Risk scores and profiles per analysis run
  - markov_results   : State transition predictions
"""

import sqlite3
import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional


# Default database file
DB_PATH = "arthsetu_data.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # Better concurrent access
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_database(db_path: str = DB_PATH) -> None:
    """
    Create all necessary tables if they don't exist.
    Safe to call multiple times (idempotent).
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # ── Transactions Table ────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            date            TEXT NOT NULL,
            description     TEXT NOT NULL,
            amount          REAL NOT NULL,
            category        TEXT NOT NULL,
            confidence      REAL DEFAULT 0,
            abs_amount      REAL NOT NULL,
            is_expense      INTEGER DEFAULT 1,
            month_str       TEXT,
            day_of_week     TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Category Summary Table ────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS category_summary (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            category        TEXT NOT NULL,
            txn_count       INTEGER DEFAULT 0,
            total_amount    REAL DEFAULT 0,
            avg_amount      REAL DEFAULT 0,
            percentage      REAL DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Risk Assessments Table ────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_assessments (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT UNIQUE NOT NULL,
            filename            TEXT,
            total_income        REAL,
            total_expense       REAL,
            net_savings         REAL,
            total_emi           REAL,
            total_investment    REAL,
            expense_income_ratio REAL,
            savings_rate        REAL,
            debt_income_ratio   REAL,
            investment_rate     REAL,
            risk_score          REAL,
            risk_level          TEXT,
            insights_json       TEXT,
            risk_factors_json   TEXT,
            positive_factors_json TEXT,
            created_at          TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Markov Results Table ──────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS markov_results (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                  TEXT UNIQUE NOT NULL,
            current_state           TEXT,
            predicted_next_state    TEXT,
            transition_probs_json   TEXT,
            forecast_states_json    TEXT,
            stationary_dist_json    TEXT,
            recommendations_json    TEXT,
            created_at              TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Analysis Runs Table ───────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_runs (
            run_id          TEXT PRIMARY KEY,
            filename        TEXT,
            row_count       INTEGER,
            date_range      TEXT,
            status          TEXT DEFAULT 'completed',
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Indexes for faster queries ────────────────────────────────────────────
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_run ON transactions(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_category ON transactions(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_run ON risk_assessments(run_id)")

    conn.commit()
    conn.close()
    print(f"   ✅ Database initialized: {db_path}")


def generate_run_id(filename: str = "") -> str:
    """Generate a unique run ID for this analysis session."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(filename))[0][:20] if filename else "run"
    return f"{base}_{ts}"


def save_transactions(
    df: pd.DataFrame,
    run_id: str,
    db_path: str = DB_PATH,
) -> int:
    """
    Save classified transaction data to SQLite.

    Args:
        df: Classified DataFrame with Date, Description, Amount, Category, Confidence
        run_id: Unique identifier for this analysis run
        db_path: Path to SQLite database

    Returns:
        Number of rows inserted
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    rows_inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT INTO transactions
                    (run_id, date, description, amount, category, confidence,
                     abs_amount, is_expense, month_str, day_of_week)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                str(row["Date"].date()),
                str(row["Description"]),
                float(row["Amount"]),
                str(row["Category"]),
                float(row.get("Confidence", 0)),
                float(row["AbsAmount"]),
                int(row["Amount"] < 0),
                str(row.get("MonthStr", "")),
                str(row.get("DayOfWeek", "")),
            ))
            rows_inserted += 1
        except Exception as e:
            print(f"   ⚠️  Skipped row: {e}")

    conn.commit()
    conn.close()
    return rows_inserted


def save_category_summary(
    summary_df: pd.DataFrame,
    run_id: str,
    db_path: str = DB_PATH,
) -> None:
    """Save category summary statistics to SQLite."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    for _, row in summary_df.iterrows():
        cursor.execute("""
            INSERT INTO category_summary
                (run_id, category, txn_count, total_amount, avg_amount, percentage)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            str(row["Category"]),
            int(row["TxnCount"]),
            float(row["TotalAmount"]),
            float(row["AvgAmount"]),
            float(row["Percentage"]),
        ))

    conn.commit()
    conn.close()


def save_risk_assessment(
    risk_profile,
    run_id: str,
    filename: str = "",
    db_path: str = DB_PATH,
) -> None:
    """Save Bayesian risk assessment results to SQLite."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO risk_assessments
            (run_id, filename, total_income, total_expense, net_savings,
             total_emi, total_investment, expense_income_ratio, savings_rate,
             debt_income_ratio, investment_rate, risk_score, risk_level,
             insights_json, risk_factors_json, positive_factors_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        filename,
        risk_profile.total_income,
        risk_profile.total_expense,
        risk_profile.net_savings,
        risk_profile.total_emi,
        risk_profile.total_investment,
        risk_profile.expense_income_ratio,
        risk_profile.savings_rate,
        risk_profile.debt_income_ratio,
        risk_profile.investment_rate,
        risk_profile.risk_score,
        risk_profile.risk_level,
        json.dumps(risk_profile.insights),
        json.dumps(risk_profile.risk_factors),
        json.dumps(risk_profile.positive_factors),
    ))

    conn.commit()
    conn.close()


def save_markov_result(
    markov_result,
    run_id: str,
    db_path: str = DB_PATH,
) -> None:
    """Save Markov Chain prediction results to SQLite."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO markov_results
            (run_id, current_state, predicted_next_state,
             transition_probs_json, forecast_states_json,
             stationary_dist_json, recommendations_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        markov_result.current_state,
        markov_result.predicted_next_state,
        json.dumps(markov_result.transition_probs.tolist()),
        json.dumps(markov_result.forecast_states),
        json.dumps(markov_result.stationary_distribution.tolist()),
        json.dumps(markov_result.recommendations),
    ))

    conn.commit()
    conn.close()


def save_run_metadata(
    run_id: str,
    filename: str,
    row_count: int,
    date_range: str,
    db_path: str = DB_PATH,
) -> None:
    """Save analysis run metadata."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO analysis_runs
            (run_id, filename, row_count, date_range, status)
        VALUES (?, ?, ?, ?, 'completed')
    """, (run_id, filename, row_count, date_range))

    conn.commit()
    conn.close()


def get_all_transactions(run_id: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieve all transactions for a run as a DataFrame."""
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM transactions WHERE run_id = ? ORDER BY date",
        conn,
        params=(run_id,),
    )
    conn.close()
    return df


def get_latest_run_id(db_path: str = DB_PATH) -> Optional[str]:
    """Get the most recent run_id from the database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT run_id FROM analysis_runs ORDER BY created_at DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row["run_id"] if row else None


def get_run_history(db_path: str = DB_PATH) -> pd.DataFrame:
    """Get all historical analysis runs."""
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        """
        SELECT ar.run_id, ar.filename, ar.row_count, ar.date_range,
               ra.risk_score, ra.risk_level, ar.created_at
        FROM analysis_runs ar
        LEFT JOIN risk_assessments ra ON ar.run_id = ra.run_id
        ORDER BY ar.created_at DESC
        """,
        conn,
    )
    conn.close()
    return df


def print_database_summary(db_path: str = DB_PATH):
    """Print summary of what's stored in the database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    print(f"\n📦 Database: {db_path}")
    for table in ["transactions", "category_summary", "risk_assessments", "markov_results", "analysis_runs"]:
        cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
        cnt = cursor.fetchone()["cnt"]
        print(f"   {table:<25}: {cnt} records")

    conn.close()


if __name__ == "__main__":
    initialize_database()
    print_database_summary()
