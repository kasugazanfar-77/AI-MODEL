"""
ArthSetu AI - Transaction Classification Model
-----------------------------------------------
Uses TF-IDF vectorization + Logistic Regression (scikit-learn) to classify
bank transaction descriptions into spending categories.

Model is trained on a rich synthetic dataset that mimics real Indian bank
statement descriptions (UPI, NEFT, card payments, etc.)
"""

import numpy as np
import pandas as pd
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


# ── Category Definitions ──────────────────────────────────────────────────────
CATEGORIES = [
    "Food & Dining",
    "Transport",
    "Shopping",
    "Bills & Utilities",
    "Entertainment",
    "Healthcare",
    "Education",
    "Salary & Income",
    "Investment",
    "EMI & Loan",
    "Transfer",
    "ATM & Cash",
    "Insurance",
    "Other",
]

# ── Training Data ─────────────────────────────────────────────────────────────
# Rich labelled examples mimicking real bank statement descriptions
TRAINING_DATA = {
    "Food & Dining": [
        "swiggy order food delivery",
        "zomato restaurant payment",
        "dominos pizza online order",
        "mcdonald india pvt ltd",
        "haldirams snacks purchase",
        "starbucks coffee payment",
        "cafe coffee day upi",
        "kfc chicken meal",
        "pizza hut order online",
        "upi bigbasket grocery",
        "blinkit instant delivery",
        "dunzo grocery delivery",
        "zepto dark store delivery",
        "instamart grocery purchase",
        "milkbasket daily essentials",
        "restaurant bill payment upi",
        "hotel food canteen",
        "dabba wala tiffin service",
        "lunchbox meal delivery",
        "burger king food court",
        "subway sandwich shop",
        "chaipoint tea snacks",
        "chai sutta bar payment",
        "theobroma bakery cake",
        "naturals ice cream",
        "baskin robbins ice cream",
        "barbeque nation dinner",
        "biryani by kilo order",
        "fresh to home fish delivery",
        "licious meat delivery",
        "box8 meal combo",
        "faasos roll order",
        "behrouz biryani premium",
        "oven story pizza",
        "upi transfer swiggy food",
        "grocery store supermarket",
        "reliance smart bazaar",
        "dmart purchase groceries",
        "star bazaar hypermarket",
        "bigbazar food items",
        "more supermarket purchase",
        "nature basket organic",
        "daily needs grocery store",
        "local kirana shop payment",
        "vegetable market purchase",
        "fruit stall payment upi",
    ],
    "Transport": [
        "ola cab ride booking",
        "uber ride payment india",
        "rapido bike taxi",
        "nammayatri auto ride",
        "yulu bike rental",
        "bounce scooter rental",
        "metro card recharge delhi",
        "mumbai metro recharge",
        "bmtc bus pass renewal",
        "irctc train ticket booking",
        "indian railways ticket",
        "redbus bus ticket booking",
        "abhibus ticket booking",
        "cleartrip flight booking",
        "makemytrip travel booking",
        "goibibo flight ticket",
        "indigo airlines ticket",
        "air india flight payment",
        "spicejet ticket purchase",
        "fastag toll recharge",
        "highway toll payment",
        "petrol pump fuel filling",
        "indian oil fuel station",
        "bharat petroleum bpcl",
        "hp fuel filling station",
        "car service center payment",
        "two wheeler service",
        "vehicle insurance premium",
        "parking fee payment",
        "valet parking charges",
        "rickshaw auto payment",
        "taxi fare payment",
        "cab fare airport transfer",
        "driver salary payment",
        "vehicle emi payment",
        "car loan installment",
        "electric vehicle charging",
        "tata power ev charge",
        "ather charging station",
        "uber eats delivery fee",
        "porter truck booking",
    ],
    "Shopping": [
        "amazon india purchase",
        "flipkart order payment",
        "myntra fashion order",
        "ajio clothing purchase",
        "nykaa beauty products",
        "meesho social commerce",
        "snapdeal order payment",
        "jiomart online shopping",
        "tatacliq purchase",
        "reliance digital electronics",
        "croma electronics store",
        "vijay sales appliances",
        "decathlon sports equipment",
        "puma shoes online",
        "adidas footwear purchase",
        "nike sportswear payment",
        "zara india clothing",
        "h and m fashion",
        "westside clothing tata",
        "pantaloons retail fashion",
        "shoppers stop purchase",
        "max fashion retail",
        "lifestyle international",
        "central mall purchase",
        "palladium mall shopping",
        "phoenix mall outlet",
        "sarojini nagar market",
        "linking road shopping",
        "upi payment retail store",
        "clothes garment shop",
        "footwear shoe shop",
        "electronics gadget purchase",
        "mobile phone accessory",
        "laptop bag backpack buy",
        "home decor purchase",
        "ikea furniture india",
        "urban ladder furniture",
        "pepperfry home decor",
        "godrej interio furniture",
        "nilkamal plastics store",
        "stationery art supplies",
        "crossword bookstore",
        "amazon kindle book",
        "gift hamper purchase",
        "jewellery gold purchase",
        "tanishq jewels payment",
        "kalyan jewellers",
        "watches accessories store",
    ],
    "Bills & Utilities": [
        "electricity bill payment",
        "bescom electricity board",
        "mahadiscom bill payment",
        "tata power electricity",
        "adani electricity bill",
        "water bill municipal",
        "bwssb water charges",
        "piped gas bill payment",
        "mahanagar gas ltd",
        "indraprastha gas iggl",
        "broadband internet bill",
        "airtel fiber payment",
        "jio fiber broadband",
        "act fibernet internet",
        "hathway cable broadband",
        "mobile recharge prepaid",
        "airtel mobile topup",
        "jio recharge plan",
        "vi vodafone recharge",
        "bsnl landline bill",
        "postpaid mobile bill",
        "dth cable recharge",
        "tata sky dth payment",
        "dish tv subscription",
        "sun direct recharge",
        "municipal tax payment",
        "property tax payment bbmp",
        "society maintenance fee",
        "apartment maintenance charges",
        "rwa society payment",
        "lpg gas cylinder booking",
        "hp gas cylinder refill",
        "bharat gas booking",
        "indane gas booking",
        "housekeeping maid salary",
        "cook salary payment",
        "driver monthly salary",
        "security guard payment",
        "waste management fees",
        "water tanker supply",
        "generator fuel charges",
    ],
    "Entertainment": [
        "netflix subscription india",
        "amazon prime video",
        "disney plus hotstar",
        "sony liv subscription",
        "zee5 digital platform",
        "voot select subscription",
        "jiocinema premium",
        "mxplayer pro subscription",
        "spotify premium music",
        "apple music subscription",
        "gaana music streaming",
        "wynk music airtel",
        "youtube premium india",
        "pvr cinemas ticket",
        "inox multiplex booking",
        "carnival cinemas ticket",
        "bookmyshow movie ticket",
        "film ticket online booking",
        "live event concert ticket",
        "ipl cricket ticket",
        "dream11 fantasy sport",
        "my11circle cricket",
        "mpl mobile premier",
        "winzo gaming platform",
        "steam game purchase",
        "playstation network",
        "xbox game pass",
        "google play games",
        "ludo king in app",
        "bar and lounge payment",
        "pub nightclub entry",
        "comedy club ticket",
        "theatre play booking",
        "museum entry ticket",
        "amusement park ticket",
        "wonderla theme park",
        "snow world entry fee",
        "bowling alley payment",
        "laser tag game",
        "escape room booking",
    ],
    "Healthcare": [
        "apollo pharmacy medicine",
        "medplus health store",
        "1mg online pharmacy",
        "pharmeasy medicine order",
        "netmeds medicine delivery",
        "doctor consultation fee",
        "practo appointment booking",
        "lybrate doctor fee",
        "manipal hospital bill",
        "apollo hospital payment",
        "fortis hospital charges",
        "columbia asia hospital",
        "narayana hrudayalaya",
        "max healthcare hospital",
        "diagnostic lab test",
        "thyrocare lab payment",
        "dr lal pathlabs test",
        "metropolis lab test",
        "srl diagnostics payment",
        "blood test report fee",
        "health checkup package",
        "annual health checkup",
        "dental clinic treatment",
        "eye specialist consult",
        "optical glasses purchase",
        "lenscart eyewear order",
        "gym fitness membership",
        "cult fit membership",
        "gold gym membership",
        "yoga class fees",
        "zumba dance class",
        "physiotherapy session",
        "ayurveda treatment",
        "homeopathy medicine",
        "health supplement order",
        "protein powder purchase",
        "vitamins supplement buy",
        "sanitizer mask covid",
        "first aid kit purchase",
        "medical equipment buy",
    ],
    "Education": [
        "udemy online course",
        "coursera subscription",
        "edx course payment",
        "unacademy subscription",
        "byju learning app",
        "vedantu live class",
        "whitehat jr coding",
        "great learning course",
        "simplilearn certification",
        "upgrad edtech payment",
        "skill share subscription",
        "linkedin learning premium",
        "google workspace edu",
        "microsoft learn course",
        "aws certification exam",
        "school fee payment",
        "college tuition fees",
        "university semester fee",
        "coaching class fees",
        "tuition teacher payment",
        "ncert book purchase",
        "reference book stationery",
        "exam registration fee",
        "neet jee coaching",
        "cat mba coaching fee",
        "upsc ias coaching",
        "language learning app",
        "duolingo plus subscription",
        "speak english class",
        "music instrument class",
        "piano lessons payment",
        "art drawing class",
        "craft workshop fee",
        "hobby class payment",
        "swimming pool fee",
        "sports academy fees",
        "chess class coaching",
    ],
    "Salary & Income": [
        "salary credit neft",
        "monthly salary deposited",
        "employer salary transfer",
        "payroll salary credit",
        "wages credited account",
        "bonus payment credit",
        "increment salary hike",
        "incentive bonus credited",
        "freelance payment received",
        "consulting fee received",
        "project payment credited",
        "client payment neft",
        "invoice payment received",
        "rent income credited",
        "rental income property",
        "dividend income credited",
        "interest credited savings",
        "fd interest credited",
        "recurring deposit matured",
        "mutual fund redemption",
        "stock sale proceeds",
        "refund credited account",
        "cashback reward credited",
        "reimbursement salary",
        "travel allowance received",
        "hra house rent allowance",
        "medical allowance credit",
        "pension credited account",
        "gratuity amount received",
        "provident fund pf withdrawal",
    ],
    "Investment": [
        "mutual fund sip zerodha",
        "groww app sip investment",
        "upstox demat investment",
        "paytm money mutual fund",
        "coin zerodha direct plan",
        "sip installment investment",
        "equity mutual fund buy",
        "debt fund purchase",
        "ppf account deposit",
        "nsc national savings cert",
        "sukanya samriddhi yojana",
        "nps national pension",
        "elss tax saving fund",
        "stock market purchase",
        "share buy nse bse",
        "gold etf purchase",
        "sovereign gold bond sgb",
        "digital gold purchase",
        "real estate investment",
        "fd fixed deposit creation",
        "rd recurring deposit",
        "lic policy premium",
        "term insurance premium",
        "ulip investment plan",
        "index fund purchase",
        "etf exchange traded fund",
        "folio redemption",
        "demat account charges",
        "brokerage commission fee",
        "trading account demat",
    ],
    "EMI & Loan": [
        "home loan emi payment",
        "housing loan installment",
        "personal loan emi",
        "car loan emi payment",
        "two wheeler loan emi",
        "education loan repayment",
        "business loan emi",
        "credit card outstanding",
        "emi auto debit",
        "loan installment debit",
        "bajaj finserv emi",
        "hdfc bank loan emi",
        "icici bank emi debit",
        "sbi home loan",
        "axis bank personal loan",
        "kotak bank emi",
        "yes bank loan repay",
        "idfc bank emi payment",
        "nbfc loan emi debit",
        "microfinance loan repay",
        "consumer durable emi",
        "mobile phone emi debit",
        "laptop emi payment",
        "tv appliance emi",
        "no cost emi payment",
        "buy now pay later bnpl",
        "simpl pay later",
        "lazypay emi",
        "zest money emi",
        "early salary repayment",
    ],
    "Transfer": [
        "upi transfer payment sent",
        "neft transfer fund",
        "rtgs transfer amount",
        "imps immediate payment",
        "bank transfer inter",
        "self transfer own account",
        "wallet transfer paytm",
        "phonepe wallet debit",
        "gpay google pay transfer",
        "bhim upi payment",
        "amazon pay wallet",
        "mobikwik transfer",
        "freecharge payment",
        "airtel money transfer",
        "money sent family",
        "sent to friend upi",
        "family transfer neft",
        "rent transfer landlord",
        "advance payment vendor",
        "security deposit transfer",
        "refund transfer initiated",
        "split bill payment",
        "splitwise settlement",
        "chit fund payment",
        "rotating savings club",
    ],
    "ATM & Cash": [
        "atm cash withdrawal",
        "atm debit card cash",
        "cash withdrawal branch",
        "counter cash withdrawal",
        "pos cash withdrawal",
        "atm charges fee",
        "cash deposit machine",
        "cdm deposit cash",
        "cheque deposit clearing",
        "dd demand draft",
        "cash back pos terminal",
    ],
    "Insurance": [
        "health insurance premium",
        "mediclaim policy premium",
        "star health insurance",
        "niva bupa premium",
        "care health insurance",
        "hdfc ergo health",
        "icici lombard premium",
        "bajaj allianz insurance",
        "new india assurance",
        "lic life insurance",
        "term plan premium",
        "life cover insurance",
        "vehicle insurance premium",
        "car insurance renewal",
        "two wheeler insurance",
        "bike insurance premium",
        "home insurance property",
        "fire insurance premium",
        "travel insurance policy",
        "critical illness cover",
        "accident insurance plan",
        "group insurance employer",
    ],
    "Other": [
        "miscellaneous payment",
        "unknown transaction",
        "adjustment entry",
        "charges deducted bank",
        "service tax gst",
        "bank charges fee",
        "account maintenance",
        "sms alert charges",
        "cheque book charges",
        "duplicate statement fee",
        "locker charges annual",
        "donation ngo payment",
        "temple trust donation",
        "charity contribution",
        "political party donation",
        "crowdfunding support",
        "government fee challan",
        "rto registration fee",
        "passport fee payment",
        "visa application fee",
        "notary legal fee",
        "lawyer fees payment",
    ],
}


# ── Text Preprocessing ────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Clean and normalize transaction description text.
    Removes numbers, special chars, and normalizes whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Remove reference numbers, digits
    text = re.sub(r"\b\d+\b", "", text)
    # Remove special characters except spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Model Building ────────────────────────────────────────────────────────────
def build_training_dataset() -> tuple[list, list]:
    """
    Build training corpus from TRAINING_DATA dictionary.
    Returns (texts, labels) lists.
    """
    texts, labels = [], []
    for category, examples in TRAINING_DATA.items():
        for example in examples:
            texts.append(preprocess_text(example))
            labels.append(category)
    return texts, labels


def train_classifier(model_type: str = "logistic") -> Pipeline:
    """
    Train a TF-IDF + Classifier pipeline.

    Args:
        model_type: "logistic" for Logistic Regression, "bayes" for Naive Bayes

    Returns:
        Trained sklearn Pipeline
    """
    texts, labels = build_training_dataset()

    print(f"\n🤖 Training Transaction Classifier (TF-IDF + {model_type.title()})...")
    print(f"   📚 Training samples: {len(texts)}")
    print(f"   🏷️  Categories: {len(CATEGORIES)}")

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Choose classifier
    if model_type == "logistic":
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
    else:  # Naive Bayes
        clf = MultinomialNB(alpha=0.1)

    # Build pipeline: preprocessing → TF-IDF → Classifier
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),       # unigrams, bigrams, trigrams
                max_features=8000,
                sublinear_tf=True,        # log normalization
                min_df=1,
                strip_accents="unicode",
            ),
        ),
        ("clf", clf),
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   ✅ Model Accuracy: {accuracy * 100:.1f}%")

    # Detailed report (optional verbose)
    if accuracy < 0.70:
        print("   ⚠️  Low accuracy — consider adding more training examples")

    return pipeline


# ── Inference ─────────────────────────────────────────────────────────────────
def classify_transactions(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """
    Classify all transactions in the dataframe.

    Args:
        df: DataFrame with 'Description' column
        pipeline: Trained sklearn Pipeline

    Returns:
        DataFrame with added 'Category' and 'Confidence' columns
    """
    print("\n🔍 Classifying transactions...")

    # Preprocess descriptions
    descriptions_cleaned = df["Description"].apply(preprocess_text).tolist()

    # Predict categories
    categories = pipeline.predict(descriptions_cleaned)

    # Predict probabilities for confidence score
    proba = pipeline.predict_proba(descriptions_cleaned)
    confidence = np.max(proba, axis=1)

    df = df.copy()
    df["Category"] = categories
    df["Confidence"] = np.round(confidence * 100, 1)

    # Override: Income/salary transactions (positive amounts) → "Salary & Income"
    df.loc[df["Amount"] > 0, "Category"] = "Salary & Income"

    print(f"   ✅ Classified {len(df)} transactions")
    print(f"   📊 Avg confidence: {confidence.mean() * 100:.1f}%")

    return df


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate category-wise spending summary.

    Returns:
        DataFrame with Category, TxnCount, TotalAmount, AvgAmount, Percentage
    """
    # Only expense transactions
    expense_df = df[df["Amount"] < 0].copy()
    expense_df["AbsAmount"] = expense_df["Amount"].abs()

    total = expense_df["AbsAmount"].sum()

    summary = (
        expense_df.groupby("Category")["AbsAmount"]
        .agg(TxnCount="count", TotalAmount="sum", AvgAmount="mean")
        .reset_index()
        .sort_values("TotalAmount", ascending=False)
    )
    summary["Percentage"] = (summary["TotalAmount"] / total * 100).round(2)
    summary["TotalAmount"] = summary["TotalAmount"].round(2)
    summary["AvgAmount"] = summary["AvgAmount"].round(2)

    return summary


def print_classification_results(df: pd.DataFrame):
    """Print transaction classification results to console."""
    print("\n" + "═" * 80)
    print("  TRANSACTION CLASSIFICATION RESULTS")
    print("═" * 80)
    print(f"  {'#':<4} {'Date':<12} {'Description':<38} {'Category':<20} {'Amount':>10}")
    print("─" * 80)

    for i, row in df.iterrows():
        desc = str(row["Description"])[:36] + ".." if len(str(row["Description"])) > 36 else str(row["Description"])
        amt_str = f"₹{row['Amount']:,.0f}"
        sign = "🔴" if row["Amount"] < 0 else "💚"
        print(f"  {i+1:<4} {str(row['Date'].date()):<12} {desc:<38} {row['Category']:<20} {sign} {amt_str:>8}")

    print("═" * 80)


if __name__ == "__main__":
    # Quick test
    pipeline = train_classifier("logistic")
    test_descs = [
        "SWIGGY ORDER PAYMENT",
        "OLA CAB BOOKING",
        "AMAZON PURCHASE",
        "ELECTRICITY BILL",
        "SALARY CREDIT",
        "HOME LOAN EMI",
        "NETFLIX SUBSCRIPTION",
        "APOLLO PHARMACY",
    ]
    print("\n📝 Test Predictions:")
    for desc in test_descs:
        cleaned = preprocess_text(desc)
        pred = pipeline.predict([cleaned])[0]
        proba = pipeline.predict_proba([cleaned]).max()
        print(f"  '{desc}' → {pred} ({proba*100:.1f}%)")
