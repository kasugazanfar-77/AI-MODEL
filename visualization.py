"""
ArthSetu AI - Visualization Dashboard Module
---------------------------------------------
Generates a comprehensive multi-chart analytics dashboard using matplotlib.
Charts produced:
  1.  Expense Category Pie Chart
  2.  Bar Chart — Category Distribution (transaction count)
  3.  Monthly Expense Trend (line + area)
  4.  Risk Score Gauge (semi-circle meter)
  5.  Markov State Transition Diagram
  6.  Top Spending Categories (horizontal bar)
  7.  Spending Heatmap (day × month)
  8.  Income vs Expense Waterfall / Grouped Bar
  9.  Sub-Score Radar Chart
  10. 6-Month State Forecast Probability Chart
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works headlessly
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Arc, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")


# ── Theme ─────────────────────────────────────────────────────────────────────
THEME = {
    "bg":         "#0F0F1A",        # Dark navy background
    "surface":    "#1A1A2E",        # Card surface
    "surface2":   "#16213E",        # Lighter card
    "accent":     "#6C63FF",        # Primary purple
    "accent2":    "#00D4FF",        # Cyan accent
    "green":      "#2ECC71",
    "yellow":     "#F39C12",
    "red":        "#E74C3C",
    "text":       "#E8E8F0",
    "text_dim":   "#8888A8",
    "grid":       "#252540",
}

CATEGORY_COLORS = [
    "#6C63FF", "#00D4FF", "#2ECC71", "#F39C12", "#E74C3C",
    "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB", "#EC407A",
    "#26C6DA", "#D4E157", "#FF7043", "#AB47BC", "#42A5F5",
]

STATE_COLORS = {
    "Stable":        "#2ECC71",
    "Moderate Risk": "#F39C12",
    "High Risk":     "#E74C3C",
}


def _apply_dark_theme():
    """Apply dark theme to matplotlib globally."""
    plt.rcParams.update({
        "figure.facecolor":   THEME["bg"],
        "axes.facecolor":     THEME["surface"],
        "axes.edgecolor":     THEME["grid"],
        "axes.labelcolor":    THEME["text"],
        "axes.titlecolor":    THEME["text"],
        "xtick.color":        THEME["text_dim"],
        "ytick.color":        THEME["text_dim"],
        "text.color":         THEME["text"],
        "grid.color":         THEME["grid"],
        "grid.alpha":         0.5,
        "legend.facecolor":   THEME["surface2"],
        "legend.edgecolor":   THEME["grid"],
        "font.family":        "DejaVu Sans",
        "font.size":          10,
    })


def _title(ax, text: str, subtitle: str = ""):
    """Add styled chart title."""
    ax.set_title(text, fontsize=13, fontweight="bold", color=THEME["text"], pad=12)
    if subtitle:
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes,
            ha="center", fontsize=8, color=THEME["text_dim"],
        )


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 1: Pie Chart — Expense by Category
# ══════════════════════════════════════════════════════════════════════════════
def plot_pie_chart(ax, category_summary: pd.DataFrame):
    """Donut-style pie chart of expense distribution."""
    df = category_summary[category_summary["TotalAmount"] > 0].copy()
    df = df.sort_values("TotalAmount", ascending=False)

    # Collapse small categories into "Other"
    threshold = df["TotalAmount"].sum() * 0.02
    df_major = df[df["TotalAmount"] >= threshold]
    df_minor = df[df["TotalAmount"] < threshold]

    if not df_minor.empty:
        other_row = pd.DataFrame([{
            "Category": "Other",
            "TotalAmount": df_minor["TotalAmount"].sum(),
            "TxnCount": df_minor["TxnCount"].sum(),
            "Percentage": df_minor["Percentage"].sum(),
        }])
        df_plot = pd.concat([df_major, other_row], ignore_index=True)
    else:
        df_plot = df_major

    amounts = df_plot["TotalAmount"].values
    labels = df_plot["Category"].values
    colors = CATEGORY_COLORS[:len(labels)]

    # Explode top 2 slices slightly
    explode = [0.05 if i < 2 else 0 for i in range(len(labels))]

    wedges, texts, autotexts = ax.pie(
        amounts,
        labels=None,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops={"linewidth": 2, "edgecolor": THEME["bg"], "antialiased": True},
    )

    for at in autotexts:
        at.set_fontsize(8)
        at.set_color(THEME["text"])
        at.set_fontweight("bold")

    # Center donut hole
    centre_circle = plt.Circle((0, 0), 0.55, fc=THEME["surface"])
    ax.add_artist(centre_circle)

    # Center text
    total = amounts.sum()
    ax.text(0, 0.08, f"₹{total:,.0f}", ha="center", va="center",
            fontsize=11, fontweight="bold", color=THEME["accent2"])
    ax.text(0, -0.15, "Total Spend", ha="center", va="center",
            fontsize=8, color=THEME["text_dim"])

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors[i], label=f"{labels[i]} ({df_plot['Percentage'].iloc[i]:.1f}%)")
        for i in range(len(labels))
    ]
    ax.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7.5,
        framealpha=0.3,
    )
    _title(ax, "💸 Expense Distribution by Category")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 2: Bar Chart — Transaction Count per Category
# ══════════════════════════════════════════════════════════════════════════════
def plot_category_bar(ax, category_summary: pd.DataFrame):
    """Bar chart showing transaction count per category."""
    df = category_summary.sort_values("TxnCount", ascending=True).tail(12)
    colors = CATEGORY_COLORS[:len(df)]

    bars = ax.barh(
        df["Category"], df["TxnCount"],
        color=colors, edgecolor=THEME["bg"],
        linewidth=0.8, height=0.65,
    )

    # Value labels
    for bar, val in zip(bars, df["TxnCount"]):
        ax.text(
            bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            str(int(val)), va="center", ha="left",
            fontsize=8, color=THEME["text"],
        )

    ax.set_xlabel("Number of Transactions", color=THEME["text_dim"])
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _title(ax, "📊 Transaction Count by Category")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 3: Monthly Expense Trend Line Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_monthly_trend(ax, df: pd.DataFrame):
    """Line + area chart for monthly income vs expense trend."""
    expense_df = df[df["Amount"] < 0].copy()
    income_df  = df[df["Amount"] > 0].copy()

    # Monthly aggregation
    monthly_exp = (
        expense_df.groupby("MonthStr")["Amount"].sum().abs()
        .reset_index().rename(columns={"Amount": "Expense"})
    )
    monthly_inc = (
        income_df.groupby("MonthStr")["Amount"].sum()
        .reset_index().rename(columns={"Amount": "Income"})
    )

    # Merge and sort by proper date order
    monthly = pd.merge(monthly_exp, monthly_inc, on="MonthStr", how="outer").fillna(0)

    # Sort by date
    monthly["_sort"] = pd.to_datetime(monthly["MonthStr"], format="%b %Y", errors="coerce")
    monthly = monthly.sort_values("_sort").reset_index(drop=True)
    x = range(len(monthly))

    # Plot income
    ax.fill_between(x, monthly["Income"], alpha=0.15, color=THEME["green"])
    ax.plot(x, monthly["Income"], color=THEME["green"], linewidth=2.5,
            marker="o", markersize=5, label="Income", zorder=5)

    # Plot expense
    ax.fill_between(x, monthly["Expense"], alpha=0.15, color=THEME["red"])
    ax.plot(x, monthly["Expense"], color=THEME["red"], linewidth=2.5,
            marker="s", markersize=5, label="Expense", zorder=5)

    # Savings line
    monthly["Savings"] = monthly["Income"] - monthly["Expense"]
    ax.plot(x, monthly["Savings"].clip(lower=0), color=THEME["accent2"],
            linewidth=1.5, linestyle="--", label="Savings", zorder=4)

    # Axis setup
    ax.set_xticks(list(x))
    ax.set_xticklabels(monthly["MonthStr"].tolist(), rotation=30, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}K" if v >= 1000 else f"₹{v:.0f}")
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right")
    _title(ax, "📈 Monthly Income vs Expense Trend")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 4: Risk Score Gauge
# ══════════════════════════════════════════════════════════════════════════════
def plot_risk_gauge(ax, risk_score: float, risk_level: str):
    """Semi-circle gauge chart for risk score."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.5, 1.4)

    # Draw gauge arcs
    zones = [
        (0, 30,  THEME["green"],   "Low"),
        (30, 60, THEME["yellow"],  "Moderate"),
        (60, 100, THEME["red"],    "High"),
    ]

    for start, end, color, label in zones:
        theta1 = 180 - (start / 100 * 180)
        theta2 = 180 - (end / 100 * 180)
        arc = Arc(
            (0, 0), 2, 2,
            angle=0, theta1=theta2, theta2=theta1,
            color=color, lw=20, alpha=0.85,
        )
        ax.add_patch(arc)

        # Zone labels
        mid_angle = np.radians(180 - ((start + end) / 2 / 100 * 180))
        lx = 0.7 * np.cos(mid_angle)
        ly = 0.7 * np.sin(mid_angle)
        ax.text(lx, ly, label, ha="center", va="center",
                fontsize=7, color=THEME["text"], fontweight="bold", alpha=0.6)

    # Inner grey arc (background)
    arc_bg = Arc((0, 0), 1.6, 1.6, angle=0, theta1=0, theta2=180,
                 color=THEME["grid"], lw=12, alpha=0.5)
    ax.add_patch(arc_bg)

    # Needle
    needle_angle = np.radians(180 - (risk_score / 100 * 180))
    nx = 0.75 * np.cos(needle_angle)
    ny = 0.75 * np.sin(needle_angle)
    ax.annotate(
        "", xy=(nx, ny), xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=THEME["text"],
            lw=2.5,
            mutation_scale=15,
        ),
    )

    # Center dot
    circle = plt.Circle((0, 0), 0.06, color=THEME["text"], zorder=10)
    ax.add_artist(circle)

    # Score text
    risk_color = (
        THEME["green"] if risk_score <= 30 else
        THEME["yellow"] if risk_score <= 60 else
        THEME["red"]
    )
    ax.text(0, -0.25, f"{risk_score:.0f}", ha="center", va="center",
            fontsize=30, fontweight="bold", color=risk_color)
    ax.text(0, -0.42, "/ 100", ha="center", va="center",
            fontsize=10, color=THEME["text_dim"])
    ax.text(0, 0.15, risk_level, ha="center", va="center",
            fontsize=11, fontweight="bold", color=risk_color)

    # Tick marks
    for i in range(0, 101, 10):
        angle = np.radians(180 - (i / 100 * 180))
        r1, r2 = 0.88, 0.98
        ax.plot(
            [r1 * np.cos(angle), r2 * np.cos(angle)],
            [r1 * np.sin(angle), r2 * np.sin(angle)],
            color=THEME["text_dim"], lw=1, alpha=0.5,
        )

    _title(ax, "🎯 Risk Score Meter")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 5: Markov State Transition Diagram
# ══════════════════════════════════════════════════════════════════════════════
def plot_state_transition(ax, markov_result):
    """Visual Markov state transition diagram with arrows."""
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    states = ["Stable", "Moderate Risk", "High Risk"]
    positions = [(2, 3), (5, 3), (8, 3)]
    colors = [STATE_COLORS[s] for s in states]

    # Draw state circles
    for (x, y), state, color in zip(positions, states, colors):
        # Glow effect
        for radius, alpha in [(0.75, 0.08), (0.65, 0.12), (0.55, 0.20)]:
            glow = plt.Circle((x, y), radius, color=color, alpha=alpha, zorder=2)
            ax.add_artist(glow)

        # Main circle
        circle = plt.Circle((x, y), 0.5, color=color, zorder=3, linewidth=2,
                             edgecolor="white", alpha=0.9)
        ax.add_artist(circle)

        # State icon
        icons = {"Stable": "✓", "Moderate Risk": "~", "High Risk": "!"}
        ax.text(x, y + 0.05, icons[state], ha="center", va="center",
                fontsize=16, fontweight="bold", color="white", zorder=4)
        ax.text(x, y - 0.75, state, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=color, zorder=4)

    # Draw arrows between states
    curr_idx = ["Stable", "Moderate Risk", "High Risk"].index(markov_result.current_state)
    next_idx = ["Stable", "Moderate Risk", "High Risk"].index(markov_result.predicted_next_state)

    # Transition arrows with probabilities
    trans_matrix = markov_result.transition_matrix
    for i, (x1, y1) in enumerate(positions):
        for j, (x2, y2) in enumerate(positions):
            if i == j:
                continue
            prob = trans_matrix[i][j]
            if prob < 0.05:
                continue

            # Arrow thickness proportional to probability
            lw = 0.5 + prob * 4
            alpha = 0.3 + prob * 0.6
            color = THEME["accent"] if (i == curr_idx and j == next_idx) else THEME["text_dim"]
            lw = lw * 2 if (i == curr_idx and j == next_idx) else lw

            # Slight curve
            dx, dy = x2 - x1, y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            ux, uy = dx / dist, dy / dist
            offset = 0.2 if i < j else -0.2

            ax.annotate(
                "",
                xy=(x2 - ux * 0.55, y2 - uy * 0.55 + offset),
                xytext=(x1 + ux * 0.55, y1 + uy * 0.55 + offset),
                arrowprops=dict(
                    arrowstyle=f"-|>",
                    color=color, lw=lw, alpha=alpha,
                    connectionstyle=f"arc3,rad={'0.3' if offset > 0 else '-0.3'}",
                ),
                zorder=5,
            )

            # Probability label
            mid_x = (x1 + x2) / 2 + (0.0 if i < j else 0.0)
            mid_y = (y1 + y2) / 2 + (0.5 if offset > 0 else -0.5)
            ax.text(mid_x, mid_y, f"{prob*100:.0f}%",
                    ha="center", va="center", fontsize=7,
                    color=color, alpha=alpha + 0.1,
                    bbox=dict(boxstyle="round,pad=0.1", fc=THEME["surface"], ec="none", alpha=0.6))

    # Self-loops (stay in same state)
    for i, (x, y) in enumerate(positions):
        prob = trans_matrix[i][i]
        arc = Arc((x, y + 0.5), 0.5, 0.5, angle=0, theta1=60, theta2=300,
                  color=STATE_COLORS[states[i]], lw=1.5, alpha=0.5)
        ax.add_patch(arc)
        ax.text(x, y + 1.05, f"{prob*100:.0f}%",
                ha="center", va="center", fontsize=6.5,
                color=STATE_COLORS[states[i]], alpha=0.7)

    # Highlight current → next
    ax.text(5, 5.3, f"Current: {markov_result.current_state}  →  Next: {markov_result.predicted_next_state}",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=THEME["accent2"],
            bbox=dict(boxstyle="round,pad=0.4", fc=THEME["surface2"], ec=THEME["accent"], alpha=0.8))

    _title(ax, "🔮 Markov State Transition Diagram")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 6: Top Spending Categories
# ══════════════════════════════════════════════════════════════════════════════
def plot_top_spending(ax, category_summary: pd.DataFrame):
    """Horizontal bar chart of top spending categories by amount."""
    df = category_summary.sort_values("TotalAmount", ascending=True).tail(10)
    colors = [CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i in range(len(df))]

    # Gradient effect (light to vivid)
    bars = ax.barh(
        df["Category"], df["TotalAmount"],
        color=colors, edgecolor=THEME["bg"],
        linewidth=0.8, height=0.65,
    )

    # Amount labels
    for bar, val, pct in zip(bars, df["TotalAmount"], df["Percentage"]):
        ax.text(
            bar.get_width() + df["TotalAmount"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"₹{val:,.0f}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=7.5, color=THEME["text"],
        )

    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}K")
    )
    ax.set_xlabel("Total Amount Spent", color=THEME["text_dim"])
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _title(ax, "🏆 Top Spending Categories by Amount")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 7: Spending Heatmap (Day × Month)
# ══════════════════════════════════════════════════════════════════════════════
def plot_spending_heatmap(ax, df: pd.DataFrame):
    """Heatmap showing spending pattern by day of week × month."""
    expense_df = df[df["Amount"] < 0].copy()
    expense_df["AbsAmount"] = expense_df["Amount"].abs()

    # Pivot: rows = DayOfWeek, cols = MonthStr
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    expense_df["DayOfWeek"] = pd.Categorical(expense_df["DayOfWeek"], categories=days_order, ordered=True)

    pivot = expense_df.pivot_table(
        values="AbsAmount",
        index="DayOfWeek",
        columns="MonthStr",
        aggfunc="sum",
        fill_value=0,
    )

    # Sort columns by date
    try:
        col_dates = pd.to_datetime(pivot.columns, format="%b %Y")
        sorted_cols = [c for _, c in sorted(zip(col_dates, pivot.columns))]
        pivot = pivot[sorted_cols]
    except Exception:
        pass

    # Custom colormap: dark purple → bright cyan
    cmap = LinearSegmentedColormap.from_list(
        "arthsetu",
        [THEME["surface2"], THEME["accent"], THEME["accent2"]],
    )

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Cell annotations
    if pivot.shape[0] * pivot.shape[1] <= 60:
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                if val > 0:
                    ax.text(c, r, f"₹{val/1000:.1f}K" if val >= 1000 else f"₹{val:.0f}",
                            ha="center", va="center", fontsize=6, color="white", alpha=0.85)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Amount (₹)", pad=0.02)
    _title(ax, "🗓️ Spending Heatmap (Day × Month)")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 8: Income vs Expense Grouped Bar
# ══════════════════════════════════════════════════════════════════════════════
def plot_income_vs_expense(ax, df: pd.DataFrame):
    """Grouped bar chart: monthly income vs expense with savings delta."""
    expense_df = df[df["Amount"] < 0].copy()
    income_df  = df[df["Amount"] > 0].copy()

    monthly_exp = expense_df.groupby("MonthStr")["Amount"].sum().abs()
    monthly_inc = income_df.groupby("MonthStr")["Amount"].sum()

    months = sorted(
        set(monthly_exp.index.tolist() + monthly_inc.index.tolist()),
        key=lambda m: pd.to_datetime(m, format="%b %Y", errors="coerce"),
    )

    inc_vals = [monthly_inc.get(m, 0) for m in months]
    exp_vals = [monthly_exp.get(m, 0) for m in months]
    sav_vals = [i - e for i, e in zip(inc_vals, exp_vals)]

    x = np.arange(len(months))
    width = 0.32

    ax.bar(x - width, inc_vals, width, label="Income",  color=THEME["green"],   alpha=0.85, edgecolor=THEME["bg"])
    ax.bar(x,         exp_vals, width, label="Expense", color=THEME["red"],     alpha=0.85, edgecolor=THEME["bg"])
    ax.bar(x + width, sav_vals, width, label="Savings",
           color=[THEME["accent2"] if v >= 0 else THEME["yellow"] for v in sav_vals],
           alpha=0.85, edgecolor=THEME["bg"])

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=30, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}K")
    )
    ax.axhline(0, color=THEME["text_dim"], linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _title(ax, "💰 Monthly Income vs Expense vs Savings")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 9: Risk Sub-Score Radar Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_radar_chart(ax, risk_profile):
    """Spider/radar chart of individual risk sub-scores."""
    categories = ["Expense\nRisk", "Savings\nRisk", "Debt\nRisk", "Discretionary\nRisk", "Consistency\nRisk"]
    values = [
        risk_profile.expense_risk_score,
        risk_profile.savings_risk_score,
        risk_profile.debt_risk_score,
        risk_profile.discretionary_risk_score,
        risk_profile.consistency_risk_score,
    ]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    values_plot = values + values[:1]

    ax.set_facecolor(THEME["surface"])

    # Reference circles
    for r in [20, 40, 60, 80, 100]:
        ax.plot(angles, [r] * len(angles), color=THEME["grid"], lw=0.6, alpha=0.5, linestyle="--")
        ax.text(0, r + 2, str(r), ha="center", va="bottom", fontsize=6, color=THEME["text_dim"])

    # Spoke lines
    for angle in angles[:-1]:
        ax.plot([0, angle], [0, 100], color=THEME["grid"], lw=0.5, alpha=0.4)

    # Plot area
    ax.fill(angles, values_plot, alpha=0.25, color=THEME["accent"])
    ax.plot(angles, values_plot, color=THEME["accent"], lw=2.5, marker="o", markersize=5)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8, color=THEME["text"])
    ax.set_ylim(0, 105)
    ax.set_yticks([])
    ax.spines["polar"].set_color(THEME["grid"])

    _title(ax, "🕷️ Risk Profile Radar")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 10: Markov 6-Month Forecast Probability
# ══════════════════════════════════════════════════════════════════════════════
def plot_forecast_probs(ax, markov_result):
    """Stacked area chart of state probabilities over 6-month forecast."""
    steps = len(markov_result.forecast_probs)
    x = list(range(steps))
    labels = [f"M+{i}" if i > 0 else "Now" for i in x]

    stable_probs   = [p[0] for p in markov_result.forecast_probs]
    moderate_probs = [p[1] for p in markov_result.forecast_probs]
    high_probs     = [p[2] for p in markov_result.forecast_probs]

    ax.stackplot(
        x,
        stable_probs, moderate_probs, high_probs,
        labels=["Stable", "Moderate Risk", "High Risk"],
        colors=[STATE_COLORS["Stable"], STATE_COLORS["Moderate Risk"], STATE_COLORS["High Risk"]],
        alpha=0.75,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("State Probability")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _title(ax, "🔭 6-Month Financial State Forecast")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_dashboard(
    df: pd.DataFrame,
    category_summary: pd.DataFrame,
    risk_profile,
    markov_result,
    output_path: str = "arthsetu_dashboard.png",
    show: bool = False,
):
    """
    Generate the complete ArthSetu AI analytics dashboard.

    Creates a 5×2 grid of professional charts and saves to PNG.

    Args:
        df: Classified transactions DataFrame
        category_summary: Category aggregation DataFrame
        risk_profile: RiskProfile dataclass from risk.py
        markov_result: MarkovResult dataclass from markov.py
        output_path: Where to save the PNG
        show: If True, display interactively (requires display)

    Returns:
        Path to saved PNG
    """
    print("\n🎨 Generating ArthSetu AI Analytics Dashboard...")
    _apply_dark_theme()

    # ── Figure Layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 28), dpi=120, facecolor=THEME["bg"])
    fig.subplots_adjust(hspace=0.42, wspace=0.38,
                        left=0.05, right=0.97, top=0.95, bottom=0.03)

    # Main title
    fig.text(
        0.5, 0.975,
        "🏦 ArthSetu AI — Financial Intelligence Dashboard",
        ha="center", va="top",
        fontsize=22, fontweight="bold", color=THEME["accent2"],
    )
    fig.text(
        0.5, 0.963,
        "AI-Powered Bank Statement Analysis  ·  Risk Assessment  ·  Predictive Analytics",
        ha="center", va="top",
        fontsize=11, color=THEME["text_dim"],
    )

    # Separator line
    fig.add_artist(plt.Line2D(
        [0.05, 0.95], [0.957, 0.957],
        transform=fig.transFigure,
        color=THEME["accent"], linewidth=1, alpha=0.5,
    ))

    # ── Grid Setup (5 rows, 4 cols) ───────────────────────────────────────────
    gs = GridSpec(5, 4, figure=fig, hspace=0.5, wspace=0.4,
                  left=0.05, right=0.97, top=0.955, bottom=0.03)

    # Row 0: Pie | Bar | Risk Gauge | Sub-scores radar
    ax_pie    = fig.add_subplot(gs[0, 0])
    ax_bar    = fig.add_subplot(gs[0, 1])
    ax_gauge  = fig.add_subplot(gs[0, 2])
    ax_radar  = fig.add_subplot(gs[0, 3], polar=True)

    # Row 1: Monthly trend (span 2) | Top spending (span 2)
    ax_trend  = fig.add_subplot(gs[1, :2])
    ax_top    = fig.add_subplot(gs[1, 2:])

    # Row 2: Income vs Expense (span 2) | Markov diagram (span 2)
    ax_incexp = fig.add_subplot(gs[2, :2])
    ax_markov = fig.add_subplot(gs[2, 2:])

    # Row 3: Forecast probs (span 2) | Heatmap (span 2)
    ax_forecast = fig.add_subplot(gs[3, :2])
    ax_heatmap  = fig.add_subplot(gs[3, 2:])

    # Row 4: Summary stats panel (span all)
    ax_summary = fig.add_subplot(gs[4, :])

    # ── Render Each Chart ─────────────────────────────────────────────────────
    print("   📊 Chart 1/10: Category Pie Chart...")
    plot_pie_chart(ax_pie, category_summary)

    print("   📊 Chart 2/10: Transaction Count Bar...")
    plot_category_bar(ax_bar, category_summary)

    print("   📊 Chart 3/10: Risk Gauge...")
    plot_risk_gauge(ax_gauge, risk_profile.risk_score, risk_profile.risk_level)

    print("   📊 Chart 4/10: Risk Radar...")
    plot_radar_chart(ax_radar, risk_profile)

    print("   📊 Chart 5/10: Monthly Trend...")
    plot_monthly_trend(ax_trend, df)

    print("   📊 Chart 6/10: Top Spending...")
    plot_top_spending(ax_top, category_summary)

    print("   📊 Chart 7/10: Income vs Expense...")
    plot_income_vs_expense(ax_incexp, df)

    print("   📊 Chart 8/10: Markov State Diagram...")
    plot_state_transition(ax_markov, markov_result)

    print("   📊 Chart 9/10: Forecast Probabilities...")
    plot_forecast_probs(ax_forecast, markov_result)

    print("   📊 Chart 10/10: Spending Heatmap...")
    plot_spending_heatmap(ax_heatmap, df)

    # ── Summary Stats Row ─────────────────────────────────────────────────────
    ax_summary.axis("off")
    _render_summary_panel(ax_summary, risk_profile, markov_result)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=THEME["bg"], edgecolor="none")
    print(f"\n   ✅ Dashboard saved: {output_path}")

    if show:
        matplotlib.use("Agg")
        plt.show()

    plt.close(fig)
    return output_path


def _render_summary_panel(ax, risk_profile, markov_result):
    """Render a KPI summary row at the bottom of the dashboard."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    kpis = [
        ("💰 Total Income",   f"₹{risk_profile.total_income:,.0f}",   THEME["green"]),
        ("💸 Total Expense",  f"₹{risk_profile.total_expense:,.0f}",  THEME["red"]),
        ("🏦 Net Savings",    f"₹{risk_profile.net_savings:,.0f}",
             THEME["green"] if risk_profile.net_savings >= 0 else THEME["red"]),
        ("📈 Savings Rate",   f"{risk_profile.savings_rate*100:.1f}%",
             THEME["green"] if risk_profile.savings_rate >= 0.20 else THEME["yellow"]),
        ("💳 Debt Ratio",     f"{risk_profile.debt_income_ratio*100:.1f}%",
             THEME["green"] if risk_profile.debt_income_ratio < 0.30 else THEME["red"]),
        ("🎯 Risk Score",     f"{risk_profile.risk_score:.0f}/100",
             THEME["green"] if risk_profile.risk_score <= 30 else
             THEME["yellow"] if risk_profile.risk_score <= 60 else THEME["red"]),
        ("🔮 Next State",     markov_result.predicted_next_state,
             STATE_COLORS[markov_result.predicted_next_state]),
        ("📊 Investment",     f"{risk_profile.investment_rate*100:.1f}%", THEME["accent2"]),
    ]

    n = len(kpis)
    for i, (label, value, color) in enumerate(kpis):
        x = (i + 0.5) / n

        # Background card
        rect = FancyBboxPatch(
            (i / n + 0.005, 0.05), 1 / n - 0.01, 0.9,
            boxstyle="round,pad=0.01", linewidth=1.5,
            edgecolor=color, facecolor=THEME["surface2"], alpha=0.7,
        )
        ax.add_patch(rect)

        ax.text(x, 0.70, label, ha="center", va="center",
                fontsize=8, color=THEME["text_dim"])
        ax.text(x, 0.32, value, ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)

    ax.set_title(
        "📋 Key Performance Indicators",
        fontsize=11, fontweight="bold", color=THEME["text"], pad=6,
    )


if __name__ == "__main__":
    print("visualization.py — import and call generate_dashboard() to create charts")
