"""
ArthSetu AI - Markov Chain Financial State Predictor
------------------------------------------------------
Models financial health as a Markov Chain with 3 states:
  - Stable       (Green)  → risk_score 0–30
  - Moderate Risk (Yellow) → risk_score 31–60
  - High Risk    (Red)    → risk_score 61–100

The transition matrix is calibrated using domain knowledge about
personal finance dynamics (how likely a person in each state is to
transition to another state next month).

Outputs:
  - Current state
  - Predicted next state
  - Transition probabilities
  - State trajectory for visualization
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ── State Definitions ─────────────────────────────────────────────────────────
STATES = ["Stable", "Moderate Risk", "High Risk"]
STATE_COLORS = {
    "Stable":        "#2ecc71",
    "Moderate Risk": "#f39c12",
    "High Risk":     "#e74c3c",
}
STATE_ICONS = {
    "Stable":        "🟢",
    "Moderate Risk": "🟡",
    "High Risk":     "🔴",
}


# ── Base Transition Matrix (Prior Knowledge) ──────────────────────────────────
#
# Rows = current state, Columns = next state
# Interpretation (example row 0 = Stable):
#   P(Stable → Stable)        = 0.70  (most likely to stay stable)
#   P(Stable → Moderate Risk) = 0.25  (some chance of sliding)
#   P(Stable → High Risk)     = 0.05  (unlikely to jump directly)
#
BASE_TRANSITION_MATRIX = np.array([
    # To:  Stable  Moderate  High
    [0.70,   0.25,    0.05],   # From: Stable
    [0.30,   0.50,    0.20],   # From: Moderate Risk
    [0.10,   0.35,    0.55],   # From: High Risk
])


@dataclass
class MarkovResult:
    """Result from Markov Chain state prediction."""

    current_state: str = "Stable"
    current_state_index: int = 0
    predicted_next_state: str = "Stable"
    predicted_next_index: int = 0

    transition_probs: np.ndarray = field(default_factory=lambda: np.zeros(3))
    transition_matrix: np.ndarray = field(default_factory=lambda: BASE_TRANSITION_MATRIX.copy())

    # 5-step forecast
    forecast_states: list = field(default_factory=list)
    forecast_probs: list = field(default_factory=list)

    # Stationary distribution (long-term equilibrium)
    stationary_distribution: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Explanation
    explanation: str = ""
    recommendations: list = field(default_factory=list)


def classify_state_from_score(risk_score: float) -> tuple[str, int]:
    """Map a risk score (0–100) to a Markov state."""
    if risk_score <= 30:
        return "Stable", 0
    elif risk_score <= 60:
        return "Moderate Risk", 1
    else:
        return "High Risk", 2


def adapt_transition_matrix(
    risk_score: float,
    savings_rate: float,
    expense_ratio: float,
    debt_ratio: float,
) -> np.ndarray:
    """
    Adapt the base transition matrix based on current financial signals.

    This is the Bayesian updating step — we adjust prior transition
    probabilities using observed evidence (financial ratios).

    Args:
        risk_score: Overall risk score (0–100)
        savings_rate: Net savings / income
        expense_ratio: Total expense / income
        debt_ratio: EMI / income

    Returns:
        Adapted 3×3 transition matrix
    """
    matrix = BASE_TRANSITION_MATRIX.copy()

    # ── Evidence-based adjustments ────────────────────────────────────────────

    # High savings → boost probability of staying/moving to Stable
    if savings_rate > 0.30:
        matrix[0][0] += 0.10   # More likely to stay stable
        matrix[1][0] += 0.15   # More likely to recover from moderate
        matrix[2][0] += 0.10   # Some chance of recovery from high
    elif savings_rate < 0:
        # Negative savings — increase risk of deterioration
        matrix[0][1] += 0.15
        matrix[1][2] += 0.20
        matrix[2][2] += 0.15

    # High expense ratio → increase risk transitions
    if expense_ratio > 0.90:
        matrix[0][2] += 0.10
        matrix[1][2] += 0.20
        matrix[2][2] += 0.15
    elif expense_ratio < 0.50:
        matrix[0][0] += 0.10
        matrix[1][0] += 0.10

    # High debt → increase risk
    if debt_ratio > 0.50:
        matrix[0][1] += 0.10
        matrix[1][2] += 0.15
        matrix[2][2] += 0.10
    elif debt_ratio < 0.10:
        # Low debt = stability boost
        matrix[1][0] += 0.10
        matrix[2][1] += 0.10

    # Normalize each row so probabilities sum to 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div-by-zero
    matrix = matrix / row_sums
    matrix = np.clip(matrix, 0.01, 0.99)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)  # re-normalize

    return matrix


def compute_stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the stationary (long-run) distribution of the Markov chain.
    Solves: π = π × P  subject to  sum(π) = 1

    Uses eigenvalue decomposition.
    """
    try:
        # Transpose and find eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
        # Find eigenvector corresponding to eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        stat_dist = np.real(eigenvectors[:, idx])
        stat_dist = np.abs(stat_dist)
        stat_dist = stat_dist / stat_dist.sum()
        return stat_dist
    except Exception:
        # Fallback: power iteration
        pi = np.ones(3) / 3
        for _ in range(1000):
            pi_new = pi @ matrix
            if np.allclose(pi, pi_new, atol=1e-8):
                break
            pi = pi_new
        return pi


def run_markov_forecast(
    risk_score: float,
    savings_rate: float = 0.0,
    expense_ratio: float = 0.75,
    debt_ratio: float = 0.20,
    n_steps: int = 6,
) -> MarkovResult:
    """
    Run the complete Markov Chain financial state prediction.

    Args:
        risk_score: Current risk score from Bayesian risk engine
        savings_rate: Current savings rate
        expense_ratio: Expense-to-income ratio
        debt_ratio: Debt-to-income ratio
        n_steps: Number of future months to forecast

    Returns:
        MarkovResult with state predictions and forecasts
    """
    result = MarkovResult()

    # ── Determine Current State ───────────────────────────────────────────────
    result.current_state, result.current_state_index = classify_state_from_score(risk_score)

    # ── Adapt Transition Matrix ───────────────────────────────────────────────
    result.transition_matrix = adapt_transition_matrix(
        risk_score, savings_rate, expense_ratio, debt_ratio
    )

    # ── One-Step Prediction ───────────────────────────────────────────────────
    result.transition_probs = result.transition_matrix[result.current_state_index]
    result.predicted_next_index = int(np.argmax(result.transition_probs))
    result.predicted_next_state = STATES[result.predicted_next_index]

    # ── Multi-Step Forecast ───────────────────────────────────────────────────
    state_probs = np.zeros(3)
    state_probs[result.current_state_index] = 1.0  # Start from current state (certain)

    result.forecast_states = [result.current_state]
    result.forecast_probs = [state_probs.copy()]

    for step in range(n_steps):
        state_probs = state_probs @ result.transition_matrix
        predicted_state = STATES[int(np.argmax(state_probs))]
        result.forecast_states.append(predicted_state)
        result.forecast_probs.append(state_probs.copy())

    # ── Stationary Distribution ───────────────────────────────────────────────
    result.stationary_distribution = compute_stationary_distribution(result.transition_matrix)

    # ── Generate Explanation ──────────────────────────────────────────────────
    result.explanation, result.recommendations = _generate_markov_explanation(result)

    return result


def _generate_markov_explanation(result: MarkovResult) -> tuple[str, list]:
    """Generate human-readable explanation for state transition."""
    curr = result.current_state
    nxt = result.predicted_next_state
    probs = result.transition_probs
    stat = result.stationary_distribution

    icon_curr = STATE_ICONS[curr]
    icon_next = STATE_ICONS[nxt]

    # Direction of change
    if result.predicted_next_index < result.current_state_index:
        direction = "📈 IMPROVING"
        direction_msg = "Your financial health is projected to IMPROVE next month"
    elif result.predicted_next_index > result.current_state_index:
        direction = "📉 DETERIORATING"
        direction_msg = "Your financial health may WORSEN — take corrective action now"
    else:
        direction = "→ STABLE"
        direction_msg = "Your financial state is expected to remain the same next month"

    explanation = (
        f"{direction} | {icon_curr} {curr} → {icon_next} {nxt}\n"
        f"  {direction_msg}\n"
        f"  Transition probabilities from '{curr}':\n"
        f"    → Stable       : {probs[0]*100:.1f}%\n"
        f"    → Moderate Risk: {probs[1]*100:.1f}%\n"
        f"    → High Risk    : {probs[2]*100:.1f}%\n"
        f"  Long-run equilibrium:\n"
        f"    Stable: {stat[0]*100:.1f}% | Moderate: {stat[1]*100:.1f}% | High: {stat[2]*100:.1f}%"
    )

    # Recommendations based on current state
    recommendations = []
    if curr == "Stable":
        recommendations = [
            "Maintain current saving and spending habits",
            "Consider increasing investment allocation by 5%",
            "Build 6-month emergency fund if not already done",
            "Explore tax-saving instruments (ELSS, PPF) before March",
        ]
    elif curr == "Moderate Risk":
        recommendations = [
            "Reduce discretionary spending (dining, entertainment, shopping)",
            "Prioritize EMI payments to avoid penalties",
            "Set up automated SIP for at least ₹500/month",
            "Review subscriptions — cancel unused services",
            "Target saving 20% of income starting next month",
        ]
    else:  # High Risk
        recommendations = [
            "🚨 URGENT: Create a strict monthly budget immediately",
            "Cut all non-essential expenses (OTT, dining out, shopping)",
            "Contact lender about EMI restructuring if needed",
            "Explore additional income sources (freelancing, part-time)",
            "Consult a certified financial planner",
            "Stop new credit card spending until balance is under control",
            "Set up a 52-week savings challenge starting now",
        ]

    return explanation, recommendations


def print_markov_results(result: MarkovResult):
    """Print Markov Chain analysis to console."""
    print("\n" + "═" * 60)
    print("    🔮 MARKOV CHAIN FINANCIAL STATE PREDICTOR")
    print("═" * 60)

    icon_c = STATE_ICONS[result.current_state]
    icon_n = STATE_ICONS[result.predicted_next_state]

    print(f"\n  Current State  : {icon_c}  {result.current_state}")
    print(f"  Predicted Next : {icon_n}  {result.predicted_next_state}")

    print(f"\n  Transition Probabilities (from {result.current_state}):")
    for i, state in enumerate(STATES):
        bar = "█" * int(result.transition_probs[i] * 20)
        print(f"    {STATE_ICONS[state]} {state:<15} [{bar:<20}] {result.transition_probs[i]*100:.1f}%")

    print(f"\n  📅 6-Month State Forecast:")
    for i, (state, probs) in enumerate(zip(result.forecast_states, result.forecast_probs)):
        label = "NOW  " if i == 0 else f"M+{i}  "
        icon = STATE_ICONS[state]
        print(f"    {label}: {icon} {state}")

    print(f"\n  📊 Long-run Equilibrium:")
    for i, state in enumerate(STATES):
        print(f"    {STATE_ICONS[state]} {state:<15}: {result.stationary_distribution[i]*100:.1f}%")

    print(f"\n  {result.explanation}")

    print(f"\n  💡 RECOMMENDATIONS:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"    {i}. {rec}")

    print("═" * 60)


if __name__ == "__main__":
    # Test with a moderate risk scenario
    result = run_markov_forecast(
        risk_score=55,
        savings_rate=0.12,
        expense_ratio=0.78,
        debt_ratio=0.35,
    )
    print_markov_results(result)
