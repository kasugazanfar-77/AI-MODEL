"""
ArthSetu AI - Risk Classification Engine
-----------------------------------------
Uses Bayesian probabilistic reasoning to compute a financial risk score (0–100)
based on:
  - Debt/Income ratio
  - Expense/Income ratio
  - Savings ratio
  - Category-level spending anomalies
  - Spending consistency

Risk Score interpretation:
  0–30   → Low Risk  (Green)
  31–60  → Moderate Risk (Yellow)
  61–100 → High Risk (Red)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ── Risk Thresholds (Bayesian Prior Knowledge) ────────────────────────────────
RISK_THRESHOLDS = {
    # Expense-to-Income ratio thresholds
    "expense_income": {
        "low":      0.50,    # < 50% of income on expenses = healthy
        "moderate": 0.75,    # 50-75% = moderate
        "high":     0.90,    # > 90% = high risk
    },
    # Savings rate thresholds
    "savings_rate": {
        "excellent": 0.30,   # > 30% savings = excellent
        "good":      0.20,   # 20-30% = good
        "poor":      0.10,   # 10-20% = poor
        "critical":  0.00,   # < 0%   = critical (dissaving)
    },
    # EMI/Loan to income ratio
    "debt_income": {
        "safe":     0.30,    # < 30% = RBI recommended safe
        "caution":  0.40,    # 30-40% = caution zone
        "danger":   0.50,    # > 50% = danger zone
    },
    # Entertainment + dining as % of total expense
    "discretionary": {
        "normal": 0.20,      # < 20% = normal
        "high":   0.35,      # > 35% = high discretionary
    },
    # Number of months with negative savings
    "neg_months_pct": {
        "ok":      0.10,
        "concern": 0.25,
        "alarm":   0.50,
    },
}

# Category weights for risk assessment
RISK_CATEGORY_WEIGHTS = {
    "EMI & Loan":       1.0,    # Full weight — fixed obligation
    "Bills & Utilities": 0.4,   # Necessary — moderate weight
    "Food & Dining":    0.5,
    "Transport":        0.4,
    "Shopping":         0.8,    # Discretionary — higher weight
    "Entertainment":    0.9,    # High discretionary
    "Healthcare":       0.2,    # Necessary — low weight
    "Education":        0.2,
    "Insurance":        0.1,    # Good behavior — very low weight
    "Investment":      -0.5,    # REDUCES risk (negative weight)
    "Transfer":         0.3,
    "ATM & Cash":       0.6,
    "Salary & Income":  0.0,    # Ignored
    "Other":            0.5,
}


@dataclass
class RiskProfile:
    """Holds complete risk assessment for a bank statement."""

    # Input metrics
    total_income: float = 0.0
    total_expense: float = 0.0
    total_emi: float = 0.0
    total_investment: float = 0.0
    net_savings: float = 0.0

    # Derived ratios
    expense_income_ratio: float = 0.0
    savings_rate: float = 0.0
    debt_income_ratio: float = 0.0
    investment_rate: float = 0.0
    discretionary_ratio: float = 0.0

    # Risk scores (each 0–100)
    expense_risk_score: float = 0.0
    savings_risk_score: float = 0.0
    debt_risk_score: float = 0.0
    discretionary_risk_score: float = 0.0
    consistency_risk_score: float = 0.0

    # Overall score
    risk_score: float = 0.0
    risk_level: str = "Low"
    risk_color: str = "green"

    # Insights
    insights: list = field(default_factory=list)
    positive_factors: list = field(default_factory=list)
    risk_factors: list = field(default_factory=list)

    # Monthly breakdown
    monthly_savings: dict = field(default_factory=dict)
    negative_months_count: int = 0


def compute_risk_score(df: pd.DataFrame, category_summary: pd.DataFrame) -> RiskProfile:
    """
    Compute a full Bayesian risk profile from classified transactions.

    Args:
        df: Classified transaction DataFrame (with Category, Amount)
        category_summary: Category-level spending summary

    Returns:
        RiskProfile with complete risk assessment
    """
    profile = RiskProfile()

    # ── Step 1: Extract key financial metrics ─────────────────────────────────
    income_df  = df[df["Amount"] > 0]
    expense_df = df[df["Amount"] < 0]

    profile.total_income   = income_df["Amount"].sum()
    profile.total_expense  = expense_df["Amount"].abs().sum()

    # EMI & Loan total
    emi_df = df[df["Category"] == "EMI & Loan"]
    profile.total_emi = emi_df["Amount"].abs().sum()

    # Investment total
    inv_df = df[df["Category"] == "Investment"]
    profile.total_investment = inv_df["Amount"].abs().sum()

    profile.net_savings = profile.total_income - profile.total_expense

    # Handle edge case: no income detected
    if profile.total_income <= 0:
        profile.total_income = profile.total_expense * 1.01  # prevent division by zero

    # ── Step 2: Compute Ratios ────────────────────────────────────────────────
    profile.expense_income_ratio = profile.total_expense / profile.total_income
    profile.savings_rate         = profile.net_savings / profile.total_income
    profile.debt_income_ratio    = profile.total_emi / profile.total_income
    profile.investment_rate      = profile.total_investment / profile.total_income

    # Discretionary spending (Shopping + Entertainment + Food) / Total expense
    disc_cats = ["Shopping", "Entertainment", "Food & Dining"]
    disc_total = 0
    for _, row in category_summary.iterrows():
        if row["Category"] in disc_cats:
            disc_total += row["TotalAmount"]
    profile.discretionary_ratio = disc_total / profile.total_expense if profile.total_expense > 0 else 0

    # ── Step 3: Monthly Savings Consistency ──────────────────────────────────
    if "Month" in df.columns:
        monthly = df.groupby("Month")["Amount"].sum()
        profile.monthly_savings = monthly.to_dict()
        profile.negative_months_count = (monthly < 0).sum()
    neg_months_pct = (
        profile.negative_months_count / len(profile.monthly_savings)
        if profile.monthly_savings else 0
    )

    # ── Step 4: Bayesian Scoring — Each Component ─────────────────────────────

    # 4a. Expense Risk (40% weight)
    ei = profile.expense_income_ratio
    thr = RISK_THRESHOLDS["expense_income"]
    if ei <= thr["low"]:
        profile.expense_risk_score = _smooth_risk(ei, 0, thr["low"], 0, 20)
    elif ei <= thr["moderate"]:
        profile.expense_risk_score = _smooth_risk(ei, thr["low"], thr["moderate"], 20, 55)
    else:
        profile.expense_risk_score = _smooth_risk(ei, thr["moderate"], 1.2, 55, 100)

    # 4b. Savings Risk (30% weight)
    sr = profile.savings_rate
    st = RISK_THRESHOLDS["savings_rate"]
    if sr >= st["excellent"]:
        profile.savings_risk_score = 5
    elif sr >= st["good"]:
        profile.savings_risk_score = 20
    elif sr >= st["poor"]:
        profile.savings_risk_score = 45
    elif sr >= st["critical"]:
        profile.savings_risk_score = 70
    else:
        profile.savings_risk_score = 90  # Negative savings = very high risk

    # 4c. Debt Risk (20% weight)
    di = profile.debt_income_ratio
    dt = RISK_THRESHOLDS["debt_income"]
    if di <= dt["safe"]:
        profile.debt_risk_score = _smooth_risk(di, 0, dt["safe"], 0, 25)
    elif di <= dt["caution"]:
        profile.debt_risk_score = _smooth_risk(di, dt["safe"], dt["caution"], 25, 60)
    else:
        profile.debt_risk_score = _smooth_risk(di, dt["caution"], 0.8, 60, 100)

    # 4d. Discretionary Risk (5% weight)
    dr = profile.discretionary_ratio
    discr_thr = RISK_THRESHOLDS["discretionary"]
    if dr <= discr_thr["normal"]:
        profile.discretionary_risk_score = 10
    elif dr <= discr_thr["high"]:
        profile.discretionary_risk_score = 40
    else:
        profile.discretionary_risk_score = 80

    # 4e. Consistency Risk (5% weight)
    nm = RISK_THRESHOLDS["neg_months_pct"]
    if neg_months_pct <= nm["ok"]:
        profile.consistency_risk_score = 5
    elif neg_months_pct <= nm["concern"]:
        profile.consistency_risk_score = 40
    elif neg_months_pct <= nm["alarm"]:
        profile.consistency_risk_score = 70
    else:
        profile.consistency_risk_score = 95

    # ── Step 5: Weighted Overall Score ───────────────────────────────────────
    # Bayesian weighted combination
    weights = {
        "expense":      0.35,
        "savings":      0.30,
        "debt":         0.20,
        "discretionary": 0.10,
        "consistency":  0.05,
    }

    raw_score = (
        profile.expense_risk_score      * weights["expense"] +
        profile.savings_risk_score      * weights["savings"] +
        profile.debt_risk_score         * weights["debt"] +
        profile.discretionary_risk_score * weights["discretionary"] +
        profile.consistency_risk_score  * weights["consistency"]
    )

    # Bonus: Investment reduces risk (Bayesian positive evidence)
    investment_bonus = min(15, profile.investment_rate * 50)
    raw_score = max(0, raw_score - investment_bonus)

    profile.risk_score = round(min(100, max(0, raw_score)), 1)

    # ── Step 6: Risk Level Classification ────────────────────────────────────
    if profile.risk_score <= 30:
        profile.risk_level = "Low Risk"
        profile.risk_color = "green"
    elif profile.risk_score <= 60:
        profile.risk_level = "Moderate Risk"
        profile.risk_color = "orange"
    else:
        profile.risk_level = "High Risk"
        profile.risk_color = "red"

    # ── Step 7: Generate Explainable AI Insights ─────────────────────────────
    profile.insights, profile.risk_factors, profile.positive_factors = _generate_insights(profile)

    return profile


def _smooth_risk(value: float, v_min: float, v_max: float, s_min: float, s_max: float) -> float:
    """
    Linear interpolation for smooth risk score mapping.
    Maps value in [v_min, v_max] to [s_min, s_max].
    """
    if v_max == v_min:
        return s_min
    ratio = (value - v_min) / (v_max - v_min)
    ratio = max(0, min(1, ratio))
    return s_min + ratio * (s_max - s_min)


def _generate_insights(profile: RiskProfile) -> tuple[list, list, list]:
    """
    Generate human-readable AI explanations for risk factors.
    Uses rule-based logic on computed ratios — explainability layer.
    """
    insights = []
    risk_factors = []
    positive_factors = []

    ei = profile.expense_income_ratio
    sr = profile.savings_rate
    di = profile.debt_income_ratio
    ir = profile.investment_rate
    dr = profile.discretionary_ratio

    # ── Expense Analysis ──────────────────────────────────────────────────────
    if ei > 0.90:
        msg = f"⚠️  CRITICAL: You're spending {ei*100:.0f}% of your income — almost nothing left!"
        insights.append(msg)
        risk_factors.append(f"Expense/Income ratio is dangerously high at {ei*100:.0f}%")
    elif ei > 0.75:
        msg = f"🔶 High expense ratio: {ei*100:.0f}% of income is being spent"
        insights.append(msg)
        risk_factors.append(f"High spending — {ei*100:.0f}% of income consumed")
    elif ei < 0.50:
        msg = f"✅ Healthy expense ratio: Only {ei*100:.0f}% of income spent"
        insights.append(msg)
        positive_factors.append(f"Controlled spending at {ei*100:.0f}% of income")

    # ── Savings Analysis ──────────────────────────────────────────────────────
    if sr < 0:
        msg = f"🚨 DANGER: Negative savings! You're spending ₹{abs(profile.net_savings):,.0f} MORE than you earn"
        insights.append(msg)
        risk_factors.append("Negative savings — dissaving behavior detected")
    elif sr < 0.10:
        msg = f"⚠️  Very low savings rate: {sr*100:.1f}% — financial cushion is thin"
        insights.append(msg)
        risk_factors.append(f"Savings rate is only {sr*100:.1f}% — below safe threshold of 10%")
    elif sr < 0.20:
        msg = f"🔶 Below-average savings rate: {sr*100:.1f}% (target: 20%+)"
        insights.append(msg)
    elif sr >= 0.30:
        msg = f"🌟 Excellent savings rate: {sr*100:.1f}% — you're building strong financial reserves"
        insights.append(msg)
        positive_factors.append(f"Outstanding {sr*100:.1f}% savings rate")
    else:
        msg = f"✅ Good savings rate: {sr*100:.1f}%"
        insights.append(msg)
        positive_factors.append(f"Healthy savings at {sr*100:.1f}%")

    # ── Debt Analysis ─────────────────────────────────────────────────────────
    if di > 0.50:
        msg = f"🚨 Debt burden is VERY HIGH: EMI/Loan consumes {di*100:.0f}% of income (RBI limit: 30%)"
        insights.append(msg)
        risk_factors.append(f"Debt-to-income ratio at {di*100:.0f}% — far exceeds RBI safe limit")
    elif di > 0.40:
        msg = f"⚠️  High debt burden: {di*100:.0f}% of income goes to EMIs"
        insights.append(msg)
        risk_factors.append(f"EMI burden at {di*100:.0f}% is above caution zone (40%)")
    elif di > 0.30:
        msg = f"🔶 Moderate debt: {di*100:.0f}% EMI-to-income ratio (approaching RBI limit)"
        insights.append(msg)
    elif di > 0:
        msg = f"✅ Manageable debt: {di*100:.0f}% EMI-to-income ratio (within RBI safe limit)"
        insights.append(msg)
        positive_factors.append(f"Debt is under control at {di*100:.0f}% of income")
    else:
        positive_factors.append("No EMI/loan obligations detected — debt-free")
        insights.append("✅ No EMI/loan obligations — you are debt-free")

    # ── Discretionary Spending ────────────────────────────────────────────────
    if dr > 0.40:
        msg = f"⚠️  High discretionary spending: {dr*100:.0f}% on shopping/entertainment/food"
        insights.append(msg)
        risk_factors.append(f"Discretionary spending at {dr*100:.0f}% of total expenses")
    elif dr > 0.25:
        msg = f"🔶 Moderate discretionary spending: {dr*100:.0f}% on lifestyle"
        insights.append(msg)

    # ── Investment ────────────────────────────────────────────────────────────
    if ir >= 0.15:
        msg = f"🌟 Strong investment discipline: {ir*100:.1f}% of income invested"
        insights.append(msg)
        positive_factors.append(f"Investing {ir*100:.1f}% of income — wealth building in progress")
    elif ir >= 0.05:
        msg = f"✅ Investing {ir*100:.1f}% of income — keep increasing gradually"
        insights.append(msg)
        positive_factors.append(f"Regular investments at {ir*100:.1f}% of income")
    elif ir == 0:
        msg = "⚠️  No investments detected — consider SIPs or fixed deposits"
        insights.append(msg)
        risk_factors.append("No investment activity — missing wealth creation opportunity")

    # ── Monthly Consistency ───────────────────────────────────────────────────
    if profile.negative_months_count > 0:
        msg = f"⚠️  {profile.negative_months_count} month(s) had negative net cash flow"
        insights.append(msg)
        risk_factors.append(f"Cash flow was negative in {profile.negative_months_count} months")

    return insights, risk_factors, positive_factors


def print_risk_report(profile: RiskProfile):
    """Print a formatted risk assessment report."""

    icons = {"Low Risk": "🟢", "Moderate Risk": "🟡", "High Risk": "🔴"}
    icon = icons.get(profile.risk_level, "⚪")

    print("\n" + "═" * 60)
    print("        🧠 AI RISK ASSESSMENT REPORT")
    print("═" * 60)
    print(f"  💰 Total Income   : ₹{profile.total_income:>12,.2f}")
    print(f"  💸 Total Expense  : ₹{profile.total_expense:>12,.2f}")
    print(f"  🏦 Net Savings    : ₹{profile.net_savings:>12,.2f}")
    print(f"  📊 EMI/Loan Total : ₹{profile.total_emi:>12,.2f}")
    print(f"  📈 Investments    : ₹{profile.total_investment:>12,.2f}")
    print("─" * 60)
    print(f"  📉 Expense/Income : {profile.expense_income_ratio*100:>6.1f}%")
    print(f"  🏦 Savings Rate   : {profile.savings_rate*100:>6.1f}%")
    print(f"  💳 Debt/Income    : {profile.debt_income_ratio*100:>6.1f}%")
    print(f"  📈 Investment Rate: {profile.investment_rate*100:>6.1f}%")
    print("─" * 60)
    print(f"\n  {icon}  RISK SCORE: {profile.risk_score:.1f}/100  [{profile.risk_level}]")
    print("\n  Sub-scores:")
    print(f"    Expense Risk     : {profile.expense_risk_score:.1f}/100")
    print(f"    Savings Risk     : {profile.savings_risk_score:.1f}/100")
    print(f"    Debt Risk        : {profile.debt_risk_score:.1f}/100")
    print(f"    Discretionary    : {profile.discretionary_risk_score:.1f}/100")
    print(f"    Consistency      : {profile.consistency_risk_score:.1f}/100")

    print("\n  🔍 AI INSIGHTS:")
    for insight in profile.insights:
        print(f"    {insight}")

    if profile.positive_factors:
        print("\n  💚 POSITIVE FACTORS:")
        for pf in profile.positive_factors:
            print(f"    ✔  {pf}")

    if profile.risk_factors:
        print("\n  ❗ RISK FACTORS:")
        for rf in profile.risk_factors:
            print(f"    ✘  {rf}")

    print("═" * 60)
