# AI-MODEL
# ArthSetu AI 🇮🇳
### Production-Grade Financial Intelligence Platform

> **Arth** (अर्थ) = Money / Meaning  |  **Setu** (सेतु) = Bridge

ArthSetu AI bridges the gap between raw financial transaction data and actionable intelligence using a fully ML-powered pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│  POST /classify  POST /risk  POST /predict  POST /analyze│
└──────┬──────────────┬──────────────┬──────────────┬──────┘
       │              │              │              │
  ┌────▼────┐   ┌─────▼─────┐ ┌─────▼─────┐ ┌────▼────┐
  │ model.py│   │  risk.py  │ │ markov.py │ │ utils.py│
  │         │   │           │ │           │ │         │
  │ TF-IDF  │   │ Bayesian  │ │  Markov   │ │Insights │
  │ NB + LR │   │ Beta-Bin  │ │  Chain    │ │  + Viz  │
  │ Ensemble│   │ Updating  │ │ (learned) │ │(charts) │
  └─────────┘   └───────────┘ └───────────┘ └─────────┘
```

## Project Structure

```
arthsetu/
├── app.py           # FastAPI server — all endpoints
├── model.py         # TF-IDF + Ensemble transaction classifier
├── risk.py          # Bayesian risk engine with Beta updating
├── markov.py        # Markov chain financial state predictor
├── utils.py         # Insights engine + matplotlib visualizations
├── demo.py          # Standalone full-pipeline demo
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the standalone demo (no server needed)

```bash
python demo.py
```

This will:
- Train the classifier and print accuracy metrics
- Run single + batch transaction classification
- Run Bayesian risk assessment for 3 simulated months
- Run Markov chain prediction from current state
- Generate AI insights
- Save charts to `./output_charts/`

### 3. Start the API server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Reference

### `POST /classify`

Classify one or more transaction descriptions.

**Single transaction:**
```json
{
  "transaction": "Paid Rs 300 to Swiggy"
}
```

**Response:**
```json
{
  "mode": "single",
  "result": {
    "input": "Paid Rs 300 to Swiggy",
    "category": "Food",
    "confidence": 0.89,
    "probabilities": {
      "Food": 0.89,
      "Shopping": 0.04,
      "Transport": 0.03
    }
  }
}
```

**Batch:**
```json
{
  "transactions": [
    "Paid Rs 300 to Swiggy",
    "Uber cab Rs 220",
    "HDFC SIP 5000",
    "BESCOM electricity bill"
  ]
}
```

---

### `POST /risk`

Bayesian financial risk assessment.

**Request:**
```json
{
  "monthly_income": 80000,
  "monthly_expenses": 58000,
  "monthly_debt_payments": 18000,
  "monthly_savings": 8000,
  "discretionary_spend": 15000,
  "period_label": "April 2024"
}
```

**Response includes:**
- `risk_level`: LOW / MODERATE / HIGH
- `composite_score`: weighted risk score 0–1
- `bayesian_posterior`: mean, std, 95% credible interval, α, β
- `component_scores`: individual DTI, EIR, savings, DSR scores
- `explanations`: primary driver, actionable advice

---

### `POST /predict`

Markov chain state prediction.

**Request:**
```json
{
  "current_state": "MODERATE",
  "observed_sequence": ["STABLE", "STABLE", "MODERATE", "HIGH_RISK"],
  "n_steps": 6
}
```

**Response includes:**
- `next_step_prediction`: probabilities + most likely state
- `6_month_trajectory`: step-by-step probability distributions
- `stationary_distribution`: long-run equilibrium
- `mean_first_passage_times`: expected steps between states
- `high_risk_absorption_prob_12m`
- `narrative`: human-readable explanation

---

### `POST /analyze`

**Full AI pipeline in one call.**

```json
{
  "transactions": [
    "Swiggy food Rs 350",
    "Zomato dinner Rs 420",
    "Ola cab Rs 180",
    "Netflix 649",
    "Amazon order 1499",
    "HDFC SIP 5000"
  ],
  "monthly_income": 80000,
  "monthly_expenses": 58000,
  "monthly_debt_payments": 18000,
  "monthly_savings": 8000,
  "discretionary_spend": 15000,
  "observed_states": ["STABLE", "MODERATE"],
  "include_charts": false,
  "period_label": "April 2024"
}
```

**Returns comprehensive `arthsetu_report`:**
- Health score (0–100, A–F grade)
- Full transaction analysis + distribution
- Bayesian risk assessment
- Markov state prediction + trajectory
- AI insights + recommendations
- (Optional) base64 PNG charts

---

## AI/ML Models

### 1. Transaction Classifier (`model.py`)

| Component | Implementation |
|-----------|---------------|
| Feature Extraction | TF-IDF (unigram + bigram, 5000 features, sublinear TF) |
| Classifier 1 | Multinomial Naive Bayes (α=0.3) |
| Classifier 2 | Logistic Regression (C=5.0, multinomial, lbfgs) |
| Ensemble | Soft Voting (weights: NB=1, LR=2) |
| Evaluation | 5-fold stratified cross-validation |

### 2. Risk Engine (`risk.py`)

| Component | Implementation |
|-----------|---------------|
| Model | Beta-Binomial conjugate prior Bayesian updating |
| Prior | Beta(α=1.5, β=4.0) — conservative (biased safe) |
| Posterior update | +1 to α if risky period, +1 to β if safe |
| Risk score | Weighted average: DTI(35%) + EIR(30%) + Savings(20%) + DSR(15%) |
| Output | Posterior mean, std, 95% HDI credible interval |

### 3. Markov Chain (`markov.py`)

| Component | Implementation |
|-----------|---------------|
| States | STABLE, MODERATE, HIGH_RISK |
| Learning | MLE + Laplace smoothing from observed sequences |
| Stationary dist. | Eigendecomposition of transition matrix |
| MFPT | Fundamental matrix method |
| Prediction | n-step matrix exponentiation (P^n) |

---

## Example Output

```
ArthSetu Health Score: 71/100  Grade: B (Good)

Risk Summary:
  Level       : ⚠️ MODERATE
  Probability : 45.2%  CI: [28% – 62%]
  Top Driver  : eir (Expense-to-Income Ratio)

6-Month Trajectory from MODERATE:
  Month +1: MODERATE  (S:32% M:50% H:18%)
  Month +2: MODERATE  (S:33% M:47% H:20%)
  Month +3: STABLE    (S:36% M:44% H:20%)

Recommendations:
  [HIGH]   Debt Management: DTI exceeds 40%...
  [MEDIUM] Expense Control: Expenses exceed 75%...
  [LOW]    Food Spending: Food is top category...
```

---

## License

MIT — Free for personal and commercial use.

Built with ❤️ for Indian fintech.
