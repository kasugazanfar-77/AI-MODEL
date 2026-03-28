# 🏦 ArthSetu AI — Intelligent Financial Analytics Platform

> **"Bridging the gap between data and financial wisdom"**
> 
> An AI-powered bank statement analyzer that runs **100% locally** — no internet, no cloud, no APIs.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [AI Models Explained](#-ai-models-explained)
- [Visualizations](#-visualizations)
- [Database Schema](#-database-schema)
- [Sample Excel Format](#-sample-excel-format)
- [Output Example](#-output-example)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 🧭 Overview

**ArthSetu AI** is a complete, offline-first financial intelligence system that ingests a bank statement Excel file and delivers:

- **AI-powered transaction classification** using TF-IDF + Logistic Regression
- **Bayesian risk scoring** (0–100) with explainable sub-scores
- **Markov Chain state prediction** — forecasts your financial state for the next 6 months
- **10-chart analytics dashboard** saved as a professional PNG
- **Local SQLite database** — all data stays on your machine

It is designed to feel like a real enterprise-grade financial analytics product, built entirely from standard Python libraries.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📥 Excel Reader | Auto-detects column names, handles messy data, supports `.xlsx`, `.xls`, `.csv` |
| 🤖 ML Classifier | TF-IDF vectorization + Logistic Regression — 14 spending categories |
| 🧠 Bayesian Risk | Weighted probabilistic risk score with 5 sub-dimensions |
| 🔮 Markov Chain | Adaptive transition matrix calibrated to your financial signals |
| 💡 AI Insights | Human-readable explanations — *why* risk is high, not just a score |
| 📊 10 Charts | Pie, bar, line, gauge, heatmap, radar, grouped bar, state diagram, forecast |
| 💾 SQLite DB | Full offline persistence — transactions, risk, Markov, run history |
| ⚡ Fast | Full pipeline completes in ~6 seconds on standard hardware |

---

## 📁 Project Structure

```
arthsetu_ai/
│
├── app.py                  ← Main pipeline — run this
├── excel_reader.py         ← Excel/CSV loading & validation
├── model.py                ← TF-IDF + ML transaction classifier
├── risk.py                 ← Bayesian risk scoring engine
├── markov.py               ← Markov Chain financial state predictor
├── database.py             ← SQLite local database layer
├── visualization.py        ← 10-chart matplotlib dashboard
├── generate_sample.py      ← Creates sample_bank_statement.xlsx
│
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── sample_bank_statement.xlsx   ← Auto-generated on first run
├── arthsetu_dashboard.png       ← Generated dashboard (output)
└── arthsetu_data.db             ← SQLite database (output)
```

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0.0 | Data loading, transformation, aggregation |
| `numpy` | ≥ 1.24.0 | Numerical computation, matrix operations |
| `scikit-learn` | ≥ 1.3.0 | TF-IDF, Logistic Regression, Naive Bayes |
| `matplotlib` | ≥ 3.7.0 | All chart rendering |
| `openpyxl` | ≥ 3.1.0 | Excel file reading and writing |
| `sqlite3` | built-in | Local database (no install needed) |

**No external APIs. No cloud. No internet required after install.**

---

## 📦 Installation

### Prerequisites
- Python **3.9 or higher**
- pip

### Step 1 — Clone or download the project

```bash
# If using git
git clone https://github.com/yourname/arthsetu-ai.git
cd arthsetu-ai

# Or just download and unzip the folder, then cd into it
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

That's it. No API keys. No `.env` files. No accounts.

---

## ⚡ Quick Start

```bash
# 1. Generate a sample bank statement (146 transactions, 4 months)
python generate_sample.py

# 2. Run the full AI pipeline
python app.py sample_bank_statement.xlsx --no-show

# 3. View the dashboard
open arthsetu_dashboard.png       # macOS
xdg-open arthsetu_dashboard.png   # Linux
start arthsetu_dashboard.png      # Windows
```

---

## 🚀 Usage

### Basic usage

```bash
python app.py                                 # Uses sample_bank_statement.xlsx
python app.py my_bank_statement.xlsx          # Your own file
```

### All options

```bash
python app.py [FILE] [OPTIONS]

Arguments:
  FILE                  Path to Excel/CSV bank statement (default: sample_bank_statement.xlsx)

Options:
  --model {logistic,bayes}   ML model type (default: logistic)
  --output PATH              Dashboard PNG save path (default: arthsetu_dashboard.png)
  --db PATH                  SQLite database path (default: arthsetu_data.db)
  --no-show                  Save dashboard without opening it (recommended for servers)
  --verbose                  Print full per-transaction classification table
  --help                     Show this help message
```

### Examples

```bash
# Use Naive Bayes instead of Logistic Regression
python app.py statement.xlsx --model bayes

# Custom output paths
python app.py statement.xlsx --output reports/june_dashboard.png --db data/finance.db

# Verbose mode — print every transaction with its predicted category
python app.py statement.xlsx --verbose --no-show

# Just generate the sample file
python generate_sample.py
```

---

## 🤖 AI Models Explained

### 1. Transaction Classifier — TF-IDF + Logistic Regression

**File:** `model.py`

The classifier converts transaction descriptions (like `"SWIGGY ORDER PAYMENT"`) into spending categories.

**How it works:**

```
Raw description
      │
      ▼
Text preprocessing    ← lowercase, remove numbers & special chars
      │
      ▼
TF-IDF Vectorizer     ← converts text to numerical features
  - ngram_range=(1,3)   unigrams, bigrams, trigrams
  - max_features=8000
  - sublinear_tf=True   log normalization
      │
      ▼
Logistic Regression   ← trained on 463 labelled Indian bank statement examples
      │
      ▼
Category + Confidence score
```

**14 categories supported:**

| Category | Examples |
|---|---|
| Food & Dining | Swiggy, Zomato, BigBasket, Kirana store |
| Transport | Ola, Uber, IRCTC, Petrol pump, Fastag |
| Shopping | Amazon, Flipkart, Myntra, Croma |
| Bills & Utilities | Electricity, Broadband, Gas, Mobile recharge |
| Entertainment | Netflix, Hotstar, BookMyShow, PVR |
| Healthcare | Apollo Pharmacy, Doctor fee, Gym |
| Education | Udemy, BYJU's, Coaching fees, School fees |
| Salary & Income | Salary credit, NEFT payroll, Freelance |
| Investment | Zerodha SIP, Groww, PPF, FD |
| EMI & Loan | Home loan, Car loan, Personal loan |
| Transfer | UPI transfer, NEFT, IMPS |
| ATM & Cash | ATM withdrawal, Cash deposit |
| Insurance | Health insurance, LIC, Car insurance |
| Other | Miscellaneous, bank charges |

---

### 2. Bayesian Risk Engine

**File:** `risk.py`

Computes a risk score (0–100) using weighted probabilistic reasoning across 5 dimensions.

**Risk Formula:**

```
Risk Score = (Expense Risk × 0.35)
           + (Savings Risk × 0.30)
           + (Debt Risk    × 0.20)
           + (Discretionary Risk × 0.10)
           + (Consistency Risk   × 0.05)
           - Investment Bonus (up to 15 points)
```

**Risk Thresholds:**

| Metric | Low | Moderate | High |
|---|---|---|---|
| Expense/Income | < 50% | 50–75% | > 90% |
| Savings Rate | > 30% | 20–30% | < 10% |
| Debt/Income | < 30% | 30–40% | > 50% |

**Risk Levels:**

| Score | Level | Indicator |
|---|---|---|
| 0–30 | Low Risk | 🟢 Green |
| 31–60 | Moderate Risk | 🟡 Yellow |
| 61–100 | High Risk | 🔴 Red |

---

### 3. Markov Chain State Predictor

**File:** `markov.py`

Models your financial health as a 3-state Markov Chain and forecasts transitions month-by-month.

**States:**
- 🟢 **Stable** — risk score 0–30
- 🟡 **Moderate Risk** — risk score 31–60
- 🔴 **High Risk** — risk score 61–100

**Base Transition Matrix (prior knowledge):**

```
                  → Stable   → Moderate  → High Risk
From Stable    :    70%         25%          5%
From Moderate  :    30%         50%         20%
From High Risk :    10%         35%         55%
```

**Bayesian Adaptation:** The matrix is dynamically adjusted based on your actual savings rate, expense ratio, and debt ratio — so predictions reflect *your* specific financial trajectory.

**Outputs:**
- Current state
- Predicted next state (most probable)
- Transition probabilities (all 3)
- 6-month probabilistic forecast
- Long-run stationary equilibrium
- Personalised action recommendations

---

## 📊 Visualizations

The dashboard generates **10 professional charts** in a single PNG:

| # | Chart | Description |
|---|---|---|
| 1 | 🥧 Donut Pie Chart | Expense distribution by category (% of total) |
| 2 | 📊 Horizontal Bar | Transaction count per category |
| 3 | 🎯 Risk Gauge | Semi-circle meter showing risk score 0–100 |
| 4 | 🕷️ Radar Chart | Spider chart of 5 risk sub-scores |
| 5 | 📈 Monthly Trend | Line + area chart — income vs expense over time |
| 6 | 🏆 Top Spending | Horizontal bar — highest expense categories with ₹ amounts |
| 7 | 💰 Grouped Bar | Monthly income vs expense vs savings comparison |
| 8 | 🔮 State Diagram | Markov transition diagram with arrow probabilities |
| 9 | 🔭 Forecast Chart | Stacked area — 6-month state probability evolution |
| 10 | 🗓️ Heatmap | Spending intensity by day-of-week × month |

All charts use a **dark navy theme** with purple/cyan accents for a professional, modern look.

---

## 🗄️ Database Schema

**File:** `arthsetu_data.db` (SQLite, auto-created)

### `transactions`
```sql
id, run_id, date, description, amount, category, confidence,
abs_amount, is_expense, month_str, day_of_week, created_at
```

### `category_summary`
```sql
id, run_id, category, txn_count, total_amount, avg_amount, percentage, created_at
```

### `risk_assessments`
```sql
id, run_id, filename, total_income, total_expense, net_savings,
total_emi, total_investment, expense_income_ratio, savings_rate,
debt_income_ratio, investment_rate, risk_score, risk_level,
insights_json, risk_factors_json, positive_factors_json, created_at
```

### `markov_results`
```sql
id, run_id, current_state, predicted_next_state,
transition_probs_json, forecast_states_json,
stationary_dist_json, recommendations_json, created_at
```

### `analysis_runs`
```sql
run_id, filename, row_count, date_range, status, created_at
```

---

## 📄 Sample Excel Format

Your bank statement Excel file should have these **3 columns** (column names are auto-detected, case-insensitive):

| Date | Description | Amount |
|---|---|---|
| 01-Jan-2024 | SALARY CREDIT NEFT INFOSYS | 85000 |
| 02-Jan-2024 | HOME LOAN EMI HDFC BANK | -21500 |
| 03-Jan-2024 | SWIGGY ORDER PAYMENT UPI | -450 |
| 05-Jan-2024 | ZERODHA SIP MUTUAL FUND | -5000 |
| 08-Jan-2024 | AIRTEL FIBER BROADBAND BILL | -999 |
| 10-Jan-2024 | OLA CAB BOOKING UPI | -180 |
| 12-Jan-2024 | AMAZON INDIA PURCHASE | -3499 |
| 15-Jan-2024 | BESCOM ELECTRICITY BILL | -1850 |

**Amount sign convention:**
- **Positive** → Income / Credit (salary, refunds)
- **Negative** → Expense / Debit (spending, EMIs, investments)

**Accepted date formats:** `DD-MM-YYYY`, `YYYY-MM-DD`, `DD/MM/YYYY`, `01 Jan 2024`, etc.

**Auto-detected column name aliases:**

| Standard Name | Also recognized as |
|---|---|
| Date | txn date, transaction date, value date |
| Description | narration, particulars, remarks, details |
| Amount | debit, credit, withdrawal, deposit |

---

## 📋 Output Example

```
╔══════════════════════════════════════════════════════════╗
║          ArthSetu AI — Financial Intelligence Platform   ║
╚══════════════════════════════════════════════════════════╝

  ⏱️   Total time       : 6.0 seconds
  📊  Transactions     : 146
  🏷️   Categories       : 11
  🎯  Risk Score       : 71.8/100 [High Risk]
  🔮  Financial State  : High Risk → High Risk
  💸  Total Expense    : ₹5,32,736.69
  🏦  Net Savings      : ₹-1,81,763.07
  📈  Savings Rate     : -51.8%
  📊  Dashboard saved  : arthsetu_dashboard.png
  💾  Database         : arthsetu_data.db

  💡 TOP AI RECOMMENDATIONS:
    1. 🚨 URGENT: Create a strict monthly budget immediately
    2. Cut all non-essential expenses (OTT, dining out, shopping)
    3. Contact lender about EMI restructuring if needed
    4. Explore additional income sources (freelancing, part-time)
    5. Consult a certified financial planner
```

---

## ⚙️ Configuration

No config files needed. All options are passed via CLI arguments. 

To generate a custom sample statement with different parameters, edit `generate_sample.py`:

```python
generate_sample_statement(
    months=6,               # How many months of history
    monthly_income=120000,  # Approximate monthly salary (₹)
    output_path="my_test.xlsx"
)
```

To add new transaction training examples, edit the `TRAINING_DATA` dict in `model.py`:

```python
TRAINING_DATA = {
    "Food & Dining": [
        "new merchant name here",   # ← add your examples
        ...
    ],
    ...
}
```

---

## 🔧 Troubleshooting

### `FileNotFoundError: sample_bank_statement.xlsx`
Run `python generate_sample.py` first to create the sample file.

### `ModuleNotFoundError: No module named 'sklearn'`
Run `pip install -r requirements.txt` again.

### Charts look blank / `_tkinter.TclError`
Add `--no-show` flag. The dashboard is always saved as PNG regardless:
```bash
python app.py statement.xlsx --no-show
```

### Excel file not reading correctly
Ensure your file has `Date`, `Description`, and `Amount` columns. Run with verbose to debug:
```bash
python app.py statement.xlsx --verbose
```

### Low ML accuracy warning
This is normal for small datasets. The model still classifies transactions meaningfully. To improve accuracy, add more training examples to `TRAINING_DATA` in `model.py`.

### `openpyxl` errors with `.xls` files
Old `.xls` format requires `xlrd`. Either convert to `.xlsx` in Excel, or install:
```bash
pip install xlrd
```

---

## 🏗️ Architecture Diagram

```
Bank Statement Excel
        │
        ▼
  excel_reader.py      ← Load, validate, clean
        │
        ▼
    model.py           ← Train TF-IDF + Logistic Regression
        │                 Classify 14 categories
        ▼
     risk.py           ← Bayesian risk score (0–100)
        │                 5 sub-dimensions + explainability
        ▼
    markov.py          ← Markov Chain state prediction
        │                 6-month forecast + recommendations
        ▼
   database.py         ← SQLite persistence (fully local)
        │
        ▼
 visualization.py      ← 10-chart matplotlib dashboard
        │
        ▼
  arthsetu_dashboard.png  +  arthsetu_data.db
```

---

## 📝 License

This project is for educational and personal use. Built with ❤️ for Indian personal finance.

---

## 🙋 Author

**ArthSetu AI** — Built as a demonstration of local AI-powered financial analytics using only standard Python ML libraries.

*"Arth" (अर्थ) = meaning / finance in Sanskrit. "Setu" (सेतु) = bridge.*
