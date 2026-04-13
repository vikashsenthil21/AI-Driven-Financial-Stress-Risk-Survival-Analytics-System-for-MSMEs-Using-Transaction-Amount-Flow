import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import urllib.parse
import plotly.express as px
import joblib

Financial_Stress_model = joblib.load("C:/Users/prave/OneDrive/Desktop/Project/models/financial_Stress_lvl/financial_stress_rf.pkl")

Financial_Stress_feature_columns = joblib.load("C:/Users/prave/OneDrive/Desktop/Project/models/financial_Stress_lvl/financial_model_features.pkl")

Survival_model = joblib.load("C:/Users/prave/OneDrive/Desktop/Project/models/xgb for survival horizon/msme_survival_horizon_model.pkl")

Survival_feature_columns = joblib.load("C:/Users/prave/OneDrive/Desktop/Project/models/xgb for survival horizon/xgbfeature_schema.pkl")

Liquidit_collapse_Model=joblib.load("C:/Users/prave/OneDrive/Desktop/Project/models/liqudity_collapse_xgb/liquidity_collapse_model.pkl")
Liquidit_collapse_features=joblib.load("C:/Users/prave/OneDrive/Desktop/Project/models/liqudity_collapse_xgb/liquidity_features.pkl")
Whatif=joblib.load("C:/Users/prave/OneDrive/Desktop/Project/models/Random forest for what if analysis/fraud_detection_model (1).pkl")
def build_model_features_for_prediction():

    conn = get_connection()

    query = """
        SELECT *
        FROM transactions
        ORDER BY txn_date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        return pd.DataFrame()

    # -------------------------------------------------
    # 1️⃣ Ensure txn_date is proper datetime (safe)
    # -------------------------------------------------
    df['txn_date'] = pd.to_datetime(df['txn_date'], errors='coerce')

    df = df.dropna(subset=['txn_date'])

    # -------------------------------------------------
    # 2️⃣ Create step (days from first transaction)
    # -------------------------------------------------
    df['step'] = (
        df['txn_date'] - df['txn_date'].min()
    ).dt.days

    # -------------------------------------------------
    # 3️⃣ Standardize type (match model training)
    # -------------------------------------------------
    df['type'] = df['txn_type'].str.upper()

    # If your model was trained on PAYMENT/TRANSFER,
    # map Credit/Debit like this:

    df['type'] = df['type'].map({
        'CREDIT': 'PAYMENT',
        'DEBIT': 'TRANSFER'
    })

    # -------------------------------------------------
    # 4️⃣ Convert amount (remove commas if needed)
    # -------------------------------------------------
    df['amount'] = (
        df['amount']
        .astype(str)
        .str.replace(',', '', regex=False)
        .astype(float)
    )

    # -------------------------------------------------
    # 5️⃣ Net amount logic
    # -------------------------------------------------
    df['net_amount'] = df.apply(
        lambda x: x['amount'] if x['txn_type'] == 'Credit' else -x['amount'],
        axis=1
    )

    # -------------------------------------------------
    # 6️⃣ Simulate balances
    # -------------------------------------------------
    df['newbalanceOrig'] = df['net_amount'].cumsum()
    df['oldbalanceOrg'] = df['newbalanceOrig'].shift(1).fillna(0)

    # -------------------------------------------------
    # 7️⃣ Dummy destination balances
    # -------------------------------------------------
    df['oldbalanceDest'] = 0
    df['newbalanceDest'] = 0

    # -------------------------------------------------
    # 8️⃣ Liquidity Ratio
    # -------------------------------------------------
    df['Liquidity_Ratio'] = (
        df['newbalanceOrig'] /
        (df['oldbalanceOrg'] + 1)
    )

    # -------------------------------------------------
    # 9️⃣ Balance Depletion
    # -------------------------------------------------
    df['Balance_Depletion'] = (
        (df['oldbalanceOrg'] - df['newbalanceOrig']) /
        (df['oldbalanceOrg'] + 1)
    )

    # -------------------------------------------------
    # 🔟 High Value Flag
    # -------------------------------------------------
    df['Is_High_Value'] = (
        df['amount'] > df['amount'].quantile(0.90)
    ).astype(int)

    # -------------------------------------------------
    # Return only model-required features
    # -------------------------------------------------
    model_features = [
        'step',
        'type',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'Liquidity_Ratio',
        'Balance_Depletion',
        'Is_High_Value'
    ]

    return df[model_features]

def predict_realtime():

    model = Whatif

    X = build_model_features_for_prediction()

    if X.empty:
        return None

    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    result = X.copy()
    result['Fraud_Probability'] = probs
    result['Fraud_Prediction'] = preds

    return result
def build_model_features():

    conn = get_connection()

    query = """
        SELECT *
        FROM transactions
        ORDER BY txn_date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        return pd.DataFrame()

    # -------------------------
    # 1️⃣ Create type column
    # -------------------------
    df['type'] = df['txn_type'].str.upper()

    # -------------------------
    # 2️⃣ Running balance (simulate oldbalanceOrg / newbalanceOrig)
    # -------------------------
    df['net_amount'] = df.apply(
        lambda x: x['amount'] if x['txn_type'] == 'Credit' else -x['amount'],
        axis=1
    )

    df['newbalanceOrig'] = df['net_amount'].cumsum()
    df['oldbalanceOrg'] = df['newbalanceOrig'].shift(1).fillna(0)

    # -------------------------
    # 3️⃣ Dummy dest balances (since MSME table has none)
    # -------------------------
    df['oldbalanceDest'] = 0
    df['newbalanceDest'] = 0

    # -------------------------
    # 4️⃣ Liquidity Ratio
    # -------------------------
    df['Liquidity_Ratio'] = df['newbalanceOrig'] / (df['oldbalanceOrg'] + 1)

    # -------------------------
    # 5️⃣ Balance Depletion
    # -------------------------
    df['Balance_Depletion'] = (
        (df['oldbalanceOrg'] - df['newbalanceOrig']) /
        (df['oldbalanceOrg'] + 1)
    )

    # -------------------------
    # 6️⃣ High Value Flag
    # -------------------------
    df['Is_High_Value'] = (df['amount'] > df['amount'].quantile(0.90)).astype(int)

    return df


def Financial_stress_risk_monthly_risk():
    conn = get_connection()

    query = """
        SELECT *
        FROM transactions
        WHERE DATE(txn_date) >= DATE('now','-30 day')
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        return None

    # ✅ FIX: Use format='mixed' to handle full datetime strings like "2026-02-17 20:49:14"
    df["txn_date"] = pd.to_datetime(df["txn_date"], format='mixed')
    df["date_only"] = df["txn_date"].dt.date

    daily_risk = []

    for date, day_df in df.groupby("date_only"):

        total_credit = day_df[day_df["txn_type"] == "Credit"]["amount"].sum()
        total_debit  = day_df[day_df["txn_type"] == "Debit"]["amount"].sum()
        total_txn    = len(day_df)

        avg_amount = day_df["amount"].mean()
        net_flow = total_credit - total_debit

        liquidity_ratio = total_credit / (total_debit + 1)
        high_value_ratio = len(day_df[day_df["amount"] > 50000]) / (total_txn + 1)
        depletion = total_debit / (total_credit + 1)

        input_data = pd.DataFrame([{
            "step": total_txn,
            "amount": avg_amount,
            "oldbalanceOrg": total_credit,
            "newbalanceOrig": net_flow,
            "oldbalanceDest": total_debit,
            "newbalanceDest": net_flow,
            "Liquidity_Ratio": liquidity_ratio,
            "Balance_Depletion": depletion,
            "Is_High_Value": high_value_ratio
        }])

        for col in Financial_Stress_feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[Financial_Stress_feature_columns]

        risk_prob = Financial_Stress_model.predict_proba(input_data)[0][1]

        daily_risk.append({
            "Date": date,
            "Risk_Probability": risk_prob
        })

    return pd.DataFrame(daily_risk)


def calculate_monthly_survival(entity_type="Company", branch=None):

    conn = get_connection()

    if entity_type == "Company":
        query = """
            SELECT 
                strftime('%Y-%m', txn_date) AS month,
                SUM(CASE WHEN txn_type='Credit' THEN amount ELSE 0 END) AS inflow,
                SUM(CASE WHEN txn_type='Debit' THEN amount ELSE 0 END) AS outflow
            FROM transactions
            WHERE entity_type='Company'
            GROUP BY month
            ORDER BY month
        """
        df = pd.read_sql(query, conn)
        current_balance = get_balance("Company")

    else:
        query = """
            SELECT 
                strftime('%Y-%m', txn_date) AS month,
                SUM(CASE WHEN txn_type='Credit' THEN amount ELSE 0 END) AS inflow,
                SUM(CASE WHEN txn_type='Debit' THEN amount ELSE 0 END) AS outflow
            FROM transactions
            WHERE entity_type='Branch' AND branch=?
            GROUP BY month
            ORDER BY month
        """
        df = pd.read_sql(query, conn, params=(branch,))
        current_balance = get_balance("Branch", branch)

    conn.close()

    if df.empty:
        return None

    survival_predictions = []

    for i in range(len(df)):

        inflow = df.loc[i, "inflow"]
        outflow = df.loc[i, "outflow"]

        monthly_burn = outflow - inflow
        avg_txn = (inflow + outflow) / 2

        features = pd.DataFrame([{
            "Current_Balance": current_balance,
            "Avg_Transaction_Amount": avg_txn,
            "Avg_Balance_Depletion": monthly_burn / (current_balance + 1e-6),
            "Avg_Liquidity_Ratio": current_balance / (outflow + 1e-6),
            "High_Value_Ratio": 0.0,
            "Active_Months": len(df),
            "Monthly_Burn": monthly_burn
        }])

        features = features[Survival_feature_columns]

        prediction = Survival_model.predict(features)[0]

        survival_predictions.append(prediction)

    df["survival_months"] = survival_predictions

    return df



def get_connection():
    return sqlite3.connect(
        "C:/Users/prave/OneDrive/Desktop/Project/metabase/finguard.db",
        check_same_thread=False
    )

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_date TEXT,
            entity_type TEXT,
            branch TEXT,
            from_entity TEXT,
            to_entity TEXT,
            amount REAL,
            txn_type TEXT,
            category TEXT
        )

    """)
    conn.commit()
    conn.close()

init_db()

def add_transaction(entity_type, branch, from_e, to_e, amount, txn_type, category):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO transactions
        VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        entity_type,
        branch if entity_type == "Branch" else "Company",
        from_e,
        to_e,
        amount,
        txn_type,
        category
    ))
    conn.commit()
    conn.close()

import pandas as pd
def get_net_capital():
    conn = get_connection()

    query = """
        SELECT COALESCE(
            SUM(
                CASE
                    WHEN txn_type = 'Credit' THEN amount
                    WHEN txn_type = 'Debit'  THEN -amount
                    ELSE 0
                END
            ), 0
        ) AS net_capital
        FROM transactions
    """

    net_capital = pd.read_sql(query, conn).iloc[0, 0]
    conn.close()
    return float(net_capital)



def get_balance(entity_type, branch=None):
    conn = get_connection()

    if entity_type == "Company":
        query = """
            SELECT COALESCE(
                SUM(
                    CASE
                        WHEN UPPER(TRIM(txn_type)) = 'CREDIT' THEN amount
                        WHEN UPPER(TRIM(txn_type)) = 'DEBIT'  THEN -amount
                        ELSE 0
                    END
                ), 0
            ) AS balance
            FROM transactions
            WHERE UPPER(TRIM(entity_type)) = 'COMPANY'
        """
        balance = pd.read_sql(query, conn).iloc[0, 0]

    else:
        query = """
            SELECT COALESCE(
                SUM(
                    CASE
                        WHEN UPPER(TRIM(txn_type)) = 'CREDIT' THEN amount
                        WHEN UPPER(TRIM(txn_type)) = 'DEBIT'  THEN -amount
                        ELSE 0
                    END
                ), 0
            ) AS balance
            FROM transactions
            WHERE UPPER(TRIM(entity_type)) = 'BRANCH'
              AND TRIM(branch) = ?
        """
        balance = pd.read_sql(query, conn, params=(branch.strip(),)).iloc[0, 0]

    conn.close()
    return float(balance)




def fetch_transactions(entity_type=None, branch=None):
    conn = get_connection()
    query = "SELECT * FROM transactions WHERE 1=1"
    params = []
    if entity_type:
        query += " AND entity_type=?"
        params.append(entity_type)
    if branch:
        query += " AND branch=?"
        params.append(branch)
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

# =============================================================================
# STREAMLIT CONFIG
# =============================================================================
st.set_page_config("FinGuard MSME", "⚙️", layout="wide")

# =============================================================================
# SIMPLE PROFESSIONAL CSS
# =============================================================================
st.markdown("""
<style>
    /* Clean background */
    .main {
        background: #FFFFFF;
        padding: 2rem;
    }
    
    /* Simple sidebar */
    [data-testid="stSidebar"] {
        background: #F8F9FA;
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] button {
        width: 100%;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    /* Clean headers */
    h1, h2, h3 {
        color: #212529;
        font-weight: 600;
    }
    
    h1 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #E9ECEF;
    }
    
    h2 {
        font-size: 1.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        font-size: 1.25rem;
        margin-top: 1.5rem;
    }
    
    /* Simple metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #212529;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: #6C757D;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Clean buttons */
    .stButton button {
        background: #0D6EFD;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton button:hover {
        background: #0B5ED7;
    }
    
    /* Simple tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #DEE2E6;
        border-radius: 4px;
    }
    
    [data-testid="stDataFrame"] thead tr th {
        background: #F8F9FA;
        color: #212529;
        font-weight: 600;
        border-bottom: 2px solid #DEE2E6;
    }
    
    /* Clean forms */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border: 1px solid #CED4DA;
        border-radius: 4px;
        padding: 0.5rem;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #0D6EFD;
        outline: none;
        box-shadow: 0 0 0 0.2rem rgba(13,110,253,0.25);
    }
    
    /* Simple dividers */
    hr {
        border: none;
        border-top: 1px solid #DEE2E6;
        margin: 2rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "branch" not in st.session_state:
    st.session_state.branch = None

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ FinGuard")
    if st.button("🏠 Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()
    if st.button("💳 Transactions"):
        st.session_state.page = "Transactions"
        st.rerun()
 
    if st.button("AI Models"):
        st.session_state.page = "AIMODELS"
        st.rerun()

# =============================================================================
# METABASE DASHBOARD URLS (CHANGE IDS ONLY)
# =============================================================================
COMPANY_MB_URL = "http://localhost:3000/public/question/20a76554-4fdb-405b-90c2-fcd51dc7a946"
BRANCH_MB_URL = "http://localhost:3000/public/dashboard/a1790d25-d2dc-4180-9f92-a8bdea473c44"

def category_inflow():
    conn = get_connection()
    query = """
        SELECT category, SUM(amount) AS total_amount
        FROM transactions
        WHERE txn_type = 'Credit'
        GROUP BY category
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df



def category_wise_outflow():
    conn = get_connection()
    query = """
        SELECT category, SUM(amount) AS total_amount
        FROM transactions
        WHERE txn_type = 'Debit'
        GROUP BY category
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# =============================================================================
# DASHBOARD (COMPANY)
# =============================================================================
if st.session_state.page == "Dashboard":

    st.title("🏢 Kovai Precision Engineering")
    st.write("Real-time Financial Intelligence Dashboard")
    st.divider()

    # Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        balance = get_balance("Company")
        st.metric("Company Net Cash", f"₹{balance:,.0f}")
    
    with col2:
        net_capital = get_net_capital()
        st.metric("Net Capital", f"₹{net_capital:,.0f}")

    # ---- COMPANY MONTHLY BOARD ----
    st.components.v1.iframe(
        COMPANY_MB_URL,
        height=500,
        scrolling=True
    )

    # ---- INFLOW DATA ----
    inflow = category_inflow()

    fig_in = px.pie(
    inflow,
    names="category",
    values="total_amount",
    hole=0.45
)

    fig_in.update_traces(
    textinfo="percent+label",
    hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}"
)
    fig_in.update_layout(title="🔻 Inflows by Category (%)")

    df_out=category_wise_outflow()
    # ---- OUTFLOW DATA ----
    fig_out = px.pie(
    df_out,
    names="category",
    values="total_amount",
    hole=0.45
)

    fig_out.update_traces(
    textinfo="percent+label",
    hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}"
)


    fig_out.update_layout(title="🔻 Outflows by Category (%)")

    # ---- SIDE-BY-SIDE DONUTS ----
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Inflows")
        st.plotly_chart(fig_in, use_container_width=True)

    with col2:
        st.markdown("### 🔻 Outflows")
        st.plotly_chart(fig_out, use_container_width=True)

    # ---- INFLOW BAR + TABLE ----
    st.subheader("💰 Inflows by Category")

    st.bar_chart(
        inflow.set_index("category")[["total_amount"]]
    )

    st.dataframe(
        inflow.style.format({
            "total_amount": "₹{:,.0f}",
            "percentage": "{:.2f}%"
        })
    )

    # Branch Navigation
    st.subheader("🏭 Branch Operations")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏢 Coimbatore HQ", use_container_width=True):
            st.session_state.page = "Branch"
            st.session_state.branch = "Coimbatore HQ"
            st.rerun()
    
    with col2:
        if st.button("🏭 Hosur Unit", use_container_width=True):
            st.session_state.page = "Branch"
            st.session_state.branch = "Hosur Unit"
            st.rerun()
    
    with col3:
        if st.button("📦 Madurai Export", use_container_width=True):
            st.session_state.page = "Branch"
            st.session_state.branch = "Madurai Export"
            st.rerun()



# =============================================================================
# TRANSACTIONS
# =============================================================================
elif st.session_state.page == "Transactions":

    st.title("💳 Transaction Management")
    st.write("Record and track all financial transactions")
    st.divider()

    st.subheader("New Transaction")
    
    with st.form("txn_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            entity_type = st.selectbox("Transaction For", ["Company", "Branch"])
            branch = "Company"
            if entity_type == "Branch":
                branch = st.selectbox("Branch", ["Coimbatore HQ", "Hosur Unit", "Madurai Export"])
            from_e = st.text_input("From")
            to_e = st.text_input("To")
        
        with col2:
            amount = st.number_input("Amount ₹", min_value=0.0)
            txn_type = st.selectbox("Type", ["Credit", "Debit"])
            category = st.selectbox("Category", ["Sales", "Salary", "Raw Material", "EB", "EMI", "GST", "Other"])

        if st.form_submit_button("Record Transaction", use_container_width=True):
            add_transaction(entity_type, branch, from_e, to_e, amount, txn_type, category)
            st.success("✅ Transaction recorded successfully")
            st.rerun()

    st.divider()
    st.subheader("Transaction History")
    st.dataframe(fetch_transactions(), use_container_width=True)

# =============================================================================
# BRANCH DASHBOARD
# =============================================================================
elif st.session_state.page == "Branch":

    branch = st.session_state.branch
    
    st.title(f"🏭 {branch}")
    st.write("Branch-level financial analytics")
    st.divider()

    balance = get_balance("Branch", branch)
    st.metric("Branch Cash Balance", f"₹{balance:,.0f}")

    # URL-encode branch filter
    branch_filter = urllib.parse.quote(branch)

    st.markdown("### 📊 Branch Analytics (Metabase)")
    st.components.v1.iframe(
    f"{BRANCH_MB_URL}?branch={branch_filter}&entity_type=Branch",
    height=800,
    scrolling=True
)


    if st.button("⬅ Back"):
        st.session_state.page = "Dashboard"
        st.rerun()

# =============================================================================
# AI MODELS PAGE
# =============================================================================
elif st.session_state.page == "AIMODELS":

    st.title("🤖 AI-Powered Analytics")
    st.write("Machine learning models for financial risk intelligence")
    st.divider()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Financial Stress Risk")
        st.write("Daily risk probability using Random Forest")
        if st.button("View Model", key="stress", use_container_width=True):
            st.session_state.page = "MODELS"
            st.rerun()
        
        st.write("")
        
    
        
        st.subheader("📊 Fraud Detection")
        st.write("Fraud Probability")
        if st.button("View Model", key="whatif", use_container_width=True):
            st.session_state.page = "Analysis"
            st.rerun()
    
    with col2:
        st.subheader("⏳ Survival Horizon")
        st.write("Predict runway with XGBoost")
        if st.button("View Model", key="surv", use_container_width=True):
            st.session_state.page = "SurvivalMODEL"
            st.rerun()
        
        st.write("")
        
        st.subheader("💧 Liquidity Collapse")
        st.write("Transaction-level risk scoring")
        if st.button("View Model", key="liq", use_container_width=True):
            st.session_state.page = "LiquidityCollapse"
            st.rerun()
        
        st.write("")
        
        

    if st.button("⬅ Back"):
        st.session_state.page = "Dashboard"
        st.rerun()


# =============================================================================
# MODELS DISPLAY PAGE
# =============================================================================
elif st.session_state.page == "MODELS":

    st.title("📈 Financial Stress Risk Model")
    st.write("**Algorithm:** Random Forest Classifier | **Window:** 30-Day Rolling")
    st.divider()

    risk_df = Financial_stress_risk_monthly_risk()

    if risk_df is not None:

        risk_df = risk_df.sort_values("Date")

        fig = px.line(
            risk_df,
            x="Date",
            y="Risk_Probability",
            markers=True,
            title="Company Financial Risk Trend (Last 30 Days)"
        )

        fig.update_layout(yaxis_tickformat=".0%")

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No transactions available for last 30 days.")

    if st.button("⬅ Back"):
        st.session_state.page = "AIMODELS"
        st.rerun()
elif st.session_state.page == "Analysis":

    st.title("🛡 Real-Time Fraud Risk Model")
    st.write("**Algorithm:** Random Forest | **Type:** Transaction-Level Fraud Detection")
    st.divider()

    fraud_df = predict_realtime()   # ← function we built earlier

    if fraud_df is not None and not fraud_df.empty:

        fraud_df = fraud_df.copy()
        fraud_df["Risk_Percentage"] = fraud_df["Fraud_Probability"] * 100

        # ----------------------------
        # 📈 Fraud Probability Trend
        # ----------------------------
        fig = px.line(
            fraud_df.reset_index(),
            x=fraud_df.index,
            y="Fraud_Probability",
            markers=True,
            title="Real-Time Fraud Risk Trend"
        )

        fig.update_layout(
            yaxis_tickformat=".0%",
            xaxis_title="Transaction Sequence",
            yaxis_title="Fraud Probability"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ----------------------------
        # 🚨 High Risk Alerts
        # ----------------------------
        high_risk = fraud_df[fraud_df["Fraud_Probability"] > 0.8]

        if not high_risk.empty:
            st.error(f"🚨 {len(high_risk)} High-Risk Transactions Detected!")
            st.dataframe(high_risk.tail(10))
        else:
            st.success("✅ No High-Risk Transactions Detected")

        st.divider()

        # ----------------------------
        # 📊 Summary Metrics
        # ----------------------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", len(fraud_df))
        col2.metric("Average Risk", f"{fraud_df['Fraud_Probability'].mean():.2%}")
        col3.metric("High Risk Count", len(high_risk))

    else:
        st.info("No transactions available for fraud analysis.")

    if st.button("⬅ Back"):
        st.session_state.page = "AIMODELS"
        st.rerun()
elif st.session_state.page == "SurvivalMODEL":

    st.title("⏳ Survival Horizon Model")
    st.write("**Algorithm:** XGBoost Regressor | **Prediction:** Runway Based on Burn Rate")
    st.divider()
    
    st.subheader("Monthly Survival Trend")

    survival_df = calculate_monthly_survival("Company")

    fig_survival = px.line(
    survival_df,
    x="month",
    y="survival_months",
    markers=True,
    title="ML-Based Survival Horizon"
)

    st.plotly_chart(fig_survival, use_container_width=True)
    
    if st.button("⬅ Back"):
        st.session_state.page = "AIMODELS"
        st.rerun()
elif st.session_state.page == "LiquidityCollapse":

    st.title("💧 Liquidity Collapse Intelligence")
    st.write("**Algorithm:** XGBoost Classifier | **Analysis:** Transaction-Level Risk Scoring")
    st.divider()

    import plotly.graph_objects as go
    import pandas as pd


    df_live = build_model_features()

    if df_live.empty:
        st.warning("No transaction data available.")
    else:

        # Predict transaction-level risk
        X_live = df_live[Liquidit_collapse_features]
        df_live['Risk_Probability'] = Liquidit_collapse_Model.predict_proba(X_live)[:, 1]

        # -----------------------------
        # Convert to Monthly Risk
        # ✅ FIX: Use format='mixed' to handle full datetime strings like "2026-02-17 20:49:14"
        # -----------------------------
        df_live['month'] = pd.to_datetime(df_live['txn_date'], format='mixed').dt.to_period('M')

        monthly = (
            df_live.groupby('month')['Risk_Probability']
            .mean()
            .reset_index()
        )

        monthly['month'] = monthly['month'].astype(str)

        # -----------------------------
        # 1️⃣ RISK TREND LINE
        # -----------------------------
        st.markdown("### 📈 Monthly Risk Trend")

        fig_line = go.Figure()

        fig_line.add_trace(go.Scatter(
            x=monthly['month'],
            y=monthly['Risk_Probability'],
            mode='lines+markers',
            name='Liquidity Risk'
        ))

        fig_line.update_layout(
            yaxis=dict(range=[0,1]),
            height=400
        )

        st.plotly_chart(fig_line, use_container_width=True)

        # -----------------------------
        # 2️⃣ CURRENT RISK GAUGE
        # -----------------------------
        st.markdown("### 🎯 Current Risk Level")

        latest_risk = monthly.iloc[-1]['Risk_Probability']

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_risk,
            number={'valueformat': '.2%'},
            gauge={
                'axis': {'range': [0, 1]},
                'steps': [
                    {'range': [0, 0.45], 'color': "green"},
                    {'range': [0.45, 0.75], 'color': "orange"},
                    {'range': [0.75, 1], 'color': "red"}
                ],
            }
        ))

        fig_gauge.update_layout(height=350)

        st.plotly_chart(fig_gauge, use_container_width=True)

        # Status Alert
        st.subheader("Current Risk Status")
        
        if latest_risk > 0.75:
            st.error(f"🚨 **CRITICAL RISK: {latest_risk:.1%}**")
            st.write("Immediate action required. Cash reserves critically low.")
        elif latest_risk > 0.45:
            st.warning(f"⚠️ **WARNING: {latest_risk:.1%}**")
            st.write("Monitor closely. Implement cash conservation measures.")
        else:
            st.success(f"✅ **STABLE: {latest_risk:.1%}**")
            st.write("Liquidity levels healthy. Continue monitoring.")
