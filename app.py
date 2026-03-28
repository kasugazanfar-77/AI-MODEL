"""
╔══════════════════════════════════════════════════════════════════════╗
║            ArthSetu AI — Intelligent Financial Analytics            ║
║         "Bridging the gap between data and financial wisdom"         ║
╚══════════════════════════════════════════════════════════════════════╝

MAIN PIPELINE — app.py
-----------------------
Orchestrates the full ArthSetu AI workflow:

  1. Load bank statement from Excel
  2. Train TF-IDF + Logistic Regression transaction classifier
  3. Classify all transactions into spending categories
  4. Compute Bayesian risk score (0–100)
  5. Run Markov Chain financial state predictor
  6. Store everything in SQLite (local, offline)
  7. Generate multi-chart analytics dashboard (PNG)
  8. Print comprehensive reports to console

Usage:
  python app.py                              → uses sample_bank_statement.xlsx
  python app.py my_statement.xlsx            → custom file
  python app.py my_statement.xlsx --no-show  → skip interactive display

Requirements:
  pip install -r requirements.txt
"""

import sys
import os
import argparse
import time
from datetime import datetime

# ── Module imports ─────────────────────────────────────────────────────────────
from excel_reader import load_bank_statement, print_summary, get_summary_stats
from model import (
    train_classifier,
    classify_transactions,
    get_category_summary,
    print_classification_results,
)
from risk import compute_risk_score, print_risk_report
from markov import run_markov_forecast, print_markov_results
from database import (
    initialize_database,
    generate_run_id,
    save_transactions,
    save_category_summary,
    save_risk_assessment,
    save_markov_result,
    save_run_metadata,
    print_database_summary,
)
from visualization import generate_dashboard


# ── ASCII Banner ────────────────────────────────────────────────────────────
BANNER = r"""
   ___         _   _      ___      _
  / _ \ _ __  | |_| |__  / __|___| |_ _  _       /\  |
 | (_) | '  \ |  _| '_ \ \__ / -_)  _| || |     /  \ |
  \___/|_|_|_| \__|_.__/ |___\___|\__|\__,_|    /____\|

         🏦 ArthSetu AI — Financial Intelligence Platform
         ─────────────────────────────────────────────────
         Powered by: ML Classification · Bayesian Risk · Markov Chains
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="ArthSetu AI — Bank Statement Financial Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="sample_bank_statement.xlsx",
        help="Path to bank statement Excel file (default: sample_bank_statement.xlsx)",
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "bayes"],
        default="logistic",
        help="ML model type: logistic regression or naive bayes (default: logistic)",
    )
    parser.add_argument(
        "--output",
        default="arthsetu_dashboard.png",
        help="Dashboard output PNG path (default: arthsetu_dashboard.png)",
    )
    parser.add_argument(
        "--db",
        default="arthsetu_data.db",
        help="SQLite database path (default: arthsetu_data.db)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't attempt to open PNG interactively (just save)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-transaction classification table",
    )
    return parser.parse_args()


def print_section(title: str):
    """Print a styled section separator."""
    width = 65
    print(f"\n{'═' * width}")
    print(f"  {'  ' + title}")
    print(f"{'═' * width}")


def run_pipeline(
    filepath: str,
    model_type: str = "logistic",
    output_path: str = "arthsetu_dashboard.png",
    db_path: str = "arthsetu_data.db",
    show_dashboard: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Execute the complete ArthSetu AI analysis pipeline.

    Args:
        filepath: Path to bank statement Excel file
        model_type: 'logistic' or 'bayes'
        output_path: Dashboard PNG save path
        db_path: SQLite database path
        show_dashboard: Whether to display dashboard interactively
        verbose: Print per-transaction table

    Returns:
        Dictionary with all analysis results
    """
    t_start = time.time()

    print(BANNER)
    print(f"  🕐 Analysis started: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    print(f"  📂 Input file: {filepath}")
    print(f"  🤖 ML model: {model_type.title()} Regression")
    print(f"  💾 Database: {db_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD BANK STATEMENT
    # ══════════════════════════════════════════════════════════════════════════
    print_section("STEP 1/6  📥 Loading Bank Statement")

    # Auto-generate sample if file doesn't exist
    if not os.path.exists(filepath) and filepath == "sample_bank_statement.xlsx":
        print("  ⚡ Sample file not found. Generating one automatically...")
        from generate_sample import generate_sample_statement
        generate_sample_statement(months=4, monthly_income=85000)

    df = load_bank_statement(filepath)
    print_summary(df)
    stats = get_summary_stats(df)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: TRAIN ML MODEL + CLASSIFY TRANSACTIONS
    # ══════════════════════════════════════════════════════════════════════════
    print_section("STEP 2/6  🤖 AI Transaction Classification")

    # Train the TF-IDF + classifier pipeline
    pipeline = train_classifier(model_type)

    # Classify all transactions
    df = classify_transactions(df, pipeline)

    # Generate category summary
    category_summary = get_category_summary(df)

    if verbose:
        print_classification_results(df)

    # Category summary table
    print(f"\n{'─' * 70}")
    print(f"  {'Category':<22} {'Txns':>5} {'Total (₹)':>14} {'Avg (₹)':>10} {'%':>7}")
    print(f"{'─' * 70}")
    for _, row in category_summary.iterrows():
        print(
            f"  {row['Category']:<22} {int(row['TxnCount']):>5} "
            f"₹{row['TotalAmount']:>12,.0f} "
            f"₹{row['AvgAmount']:>8,.0f} "
            f"{row['Percentage']:>6.1f}%"
        )
    print(f"{'─' * 70}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: BAYESIAN RISK ASSESSMENT
    # ══════════════════════════════════════════════════════════════════════════
    print_section("STEP 3/6  🧠 Bayesian Risk Assessment")

    risk_profile = compute_risk_score(df, category_summary)
    print_risk_report(risk_profile)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: MARKOV CHAIN STATE PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    print_section("STEP 4/6  🔮 Markov Chain State Predictor")

    markov_result = run_markov_forecast(
        risk_score=risk_profile.risk_score,
        savings_rate=risk_profile.savings_rate,
        expense_ratio=risk_profile.expense_income_ratio,
        debt_ratio=risk_profile.debt_income_ratio,
        n_steps=6,
    )
    print_markov_results(markov_result)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: SAVE TO LOCAL SQLite DATABASE
    # ══════════════════════════════════════════════════════════════════════════
    print_section("STEP 5/6  💾 Saving to Local Database")

    initialize_database(db_path)
    run_id = generate_run_id(filepath)

    print(f"  🔑 Run ID: {run_id}")

    n_saved = save_transactions(df, run_id, db_path)
    print(f"  ✅ Saved {n_saved} transactions")

    save_category_summary(category_summary, run_id, db_path)
    print(f"  ✅ Saved {len(category_summary)} category summaries")

    save_risk_assessment(risk_profile, run_id, filepath, db_path)
    print(f"  ✅ Saved risk assessment (score: {risk_profile.risk_score:.1f})")

    save_markov_result(markov_result, run_id, db_path)
    print(f"  ✅ Saved Markov prediction ({markov_result.current_state} → {markov_result.predicted_next_state})")

    date_range = f"{stats['date_range_start']} to {stats['date_range_end']}"
    save_run_metadata(run_id, filepath, len(df), date_range, db_path)

    print_database_summary(db_path)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6: GENERATE VISUALIZATION DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    print_section("STEP 6/6  🎨 Generating Analytics Dashboard")

    dashboard_path = generate_dashboard(
        df=df,
        category_summary=category_summary,
        risk_profile=risk_profile,
        markov_result=markov_result,
        output_path=output_path,
        show=show_dashboard,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    t_elapsed = time.time() - t_start

    print("\n" + "═" * 65)
    print("  ✅  ARTHSETU AI ANALYSIS COMPLETE")
    print("═" * 65)
    print(f"  ⏱️   Total time       : {t_elapsed:.1f} seconds")
    print(f"  📊  Transactions     : {len(df)}")
    print(f"  🏷️   Categories       : {len(category_summary)}")
    print(f"  🎯  Risk Score       : {risk_profile.risk_score:.1f}/100 [{risk_profile.risk_level}]")
    print(f"  🔮  Financial State  : {markov_result.current_state} → {markov_result.predicted_next_state}")
    print(f"  💸  Total Expense    : ₹{risk_profile.total_expense:,.2f}")
    print(f"  🏦  Net Savings      : ₹{risk_profile.net_savings:,.2f}")
    print(f"  📈  Savings Rate     : {risk_profile.savings_rate * 100:.1f}%")
    print(f"  📊  Dashboard saved  : {dashboard_path}")
    print(f"  💾  Database         : {db_path}")
    print("═" * 65)

    print("\n  💡 TOP AI RECOMMENDATIONS:")
    for i, rec in enumerate(markov_result.recommendations[:5], 1):
        print(f"    {i}. {rec}")

    print("\n" + "═" * 65)

    return {
        "df": df,
        "category_summary": category_summary,
        "risk_profile": risk_profile,
        "markov_result": markov_result,
        "run_id": run_id,
        "dashboard_path": dashboard_path,
    }


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    try:
        results = run_pipeline(
            filepath=args.file,
            model_type=args.model,
            output_path=args.output,
            db_path=args.db,
            show_dashboard=not args.no_show,
            verbose=args.verbose,
        )
    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}")
        print("   Run `python generate_sample.py` first to create a sample file.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
