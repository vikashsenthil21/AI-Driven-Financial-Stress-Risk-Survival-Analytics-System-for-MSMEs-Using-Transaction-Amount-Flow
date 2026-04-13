# AI-Driven-Financial-Stress-Risk-and-Survival-Analytics-System-Using-Transaction-Amount-Flow

AI-powered financial intelligence and risk analytics platform designed to monitor transaction flow, detect financial stress, and predict survival probability for MSMEs and multi-branch enterprises.

The system integrates Artificial Intelligence, financial modeling, and real-time analytics to provide predictive insights into liquidity risk, stress levels, anomaly detection, and business survival horizon.

By leveraging machine learning models such as XGBoost, Isolation Forest, Random Forest, and LightGBM, the platform enables proactive financial decision-making and early risk mitigation.

---

## About

The purpose of this project is to utilize Artificial Intelligence and predictive analytics to develop an intelligent financial stress and risk monitoring system. It provides real-time transaction analysis and AI-driven forecasting to help organizations:

- Detect early financial stress signals  
- Predict liquidity collapse risks  
- Estimate survival probability  
- Monitor risk acceleration patterns  
- Identify financial anomalies  

The system dynamically analyzes transaction amount flows and adapts risk scoring models to evolving financial patterns.

---

## Features

- Real-time transaction monitoring  
- Company and branch-level financial dashboards  
- AI-powered Financial Stress Level Prediction (XGBoost)  
- Liquidity Collapse Risk Detection  
- High-Risk Probability Scoring (LightGBM)  
- Risk Acceleration Modeling (XGBoost)  
- Survival Horizon Estimation  
- Financial Anomaly Detection (Isolation Forest)  
- What-if Scenario Analysis (Random Forest)  
- Category-wise inflow and outflow visualization  
- Embedded BI dashboards (Metabase integration)  

---

## AI Models Used

- **XGBoost** – Financial Stress Level Prediction  
- **XGBoost** – Risk Acceleration Detection  
- **XGBoost** – Survival Horizon Estimation  
- **LightGBM** – High-Risk Probability Prediction  
- **Isolation Forest** – Financial Anomaly Detection  
- **Random Forest** – What-if Scenario Simulation  

---

## Requirements

### Hardware Requirements

- Laptop or Desktop System  
- RAM: Minimum 4GB (8GB recommended for ML training)  
- Processor: Dual-core or higher  
- Stable Internet Connection (for dashboard embedding)  

---

### Software Requirements

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Database:** SQLite  
- **Visualization:** Plotly, Metabase  
- **Machine Learning:**  
  - XGBoost  
  - Scikit-learn  
  - LightGBM  
- **OS Support:** Windows 10+, Linux, macOS  

---

## System Architecture


<img width="1024" height="1536" alt="ChatGPT Image Feb 13, 2026, 09_16_39 PM" src="https://github.com/user-attachments/assets/9426782b-8852-4e8b-b9d0-f3cf956423b6" />


---

## Modules

### 1. Transaction Management Module
- Record credit and debit entries  
- Category tagging (Sales, Salary, GST, EMI, etc.)  
- Branch-level transaction tracking  
- Real-time balance calculation  

### 2. Financial Dashboard Module
- Company Net Capital  
- Branch Cash Balance  
- Category-wise inflow/outflow  
- Monthly performance analytics  

### 3. AI Risk Analytics Module
- Financial Stress Level Score  
- Liquidity Risk Prediction  
- Survival Probability Estimation  
- Risk Acceleration Monitoring  
- High-Risk Probability Index  

### 4. Anomaly Detection Module
- Identifies abnormal transaction patterns  
- Flags suspicious financial deviations  

---
## Sample Code:
```py
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import urllib.parse
import plotly.express as px
# =============================================================================
# DATABASE
# =============================================================================
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
    if st.button("Graph Dashboard"):
        st.session_state.page = "Graph_Dashboard"
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

    st.markdown("##### 🏢 Kovai Precision Engineering")

    # ---- METRICS ----
    balance = get_balance("Company")
    st.metric("### Company Net Cash", f"₹{balance:,.0f}")

    net_capital = get_net_capital()
    st.metric("### 💰 Net Capital", f"₹{net_capital:,.0f}")

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

    # ---- BRANCH NAVIGATION ----
    st.markdown("##### 🏭 Branches")

    branches = {
        "Coimbatore HQ": "Coimbatore HQ",
        "Hosur Unit": "Hosur Unit",
        "Madurai Export": "Madurai Export"
    }

    for label, branch in branches.items():
        if st.button(f"##### {label}"):
            st.session_state.page = "Branch"
            st.session_state.branch = branch
            st.rerun()



# =============================================================================
# TRANSACTIONS
# =============================================================================
elif st.session_state.page == "Transactions":

    st.markdown("## 💳 Transactions")

    with st.form("txn_form"):
        entity_type = st.selectbox("Transaction For", ["Company", "Branch"])
        branch = "Company"
        if entity_type == "Branch":
            branch = st.selectbox(
                "Branch",
                ["Coimbatore HQ", "Hosur Unit", "Madurai Export"]
            )

        from_e = st.text_input("From")
        to_e = st.text_input("To")
        amount = st.number_input("Amount ₹", min_value=0.0)
        txn_type = st.selectbox("Debit / Credit", ["Credit", "Debit"])
        category = st.selectbox(
            "Category",
            ["Sales", "Salary", "Raw Material", "EB", "EMI", "GST", "Other"]
        )

        if st.form_submit_button("Record Transaction"):
            add_transaction(
                entity_type, branch, from_e, to_e,
                amount, txn_type, category
            )
            st.success("Transaction Recorded")
            st.rerun()

    st.markdown("### All Transactions")
    st.dataframe(fetch_transactions(), use_container_width=True)

# =============================================================================
# BRANCH DASHBOARD
# =============================================================================
elif st.session_state.page == "Branch":

    branch = st.session_state.branch
    st.markdown(f" 🏭 {branch}")

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
elif st.session_state.page == "Graph_Dashboard":

    st.markdown("## Graph_Dashboard")
    st.components.v1.iframe(
        COMPANY_MB_URL,
        height=800,
        scrolling=True
    )
elif st.session_state.page == "AIMODELS":
    models = {
        "": "Coimbatore HQ",
        "Hosur Unit": "Hosur Unit",
        "Madurai Export": "Madurai Export"
    }

    for label, model in models.items():
        if st.button(f"##### {label}"):
            st.session_state.page = "AIMODELS"
            st.session_state.model = model
            st.rerun()

    if st.button("⬅ Back"):
        st.session_state.page = "Dashboard"
        st.rerun()
elif st.session_state.page == "models":

    model = st.session_state.models
    st.markdown(f" 🏭 {model}")

    

    

    st.markdown("### 📊 Branch Analytics (Metabase)")
    st.components.v1.iframe(
    f"{BRANCH_MB_URL}?branch={branch_filter}&entity_type=Branch",
    height=800,
    scrolling=True
)


    if st.button("⬅ Back"):
        st.session_state.page = "AIMODELS"
        st.rerun()
```
## Output

<img width="1522" height="820" alt="Screenshot 2026-02-14 172603" src="https://github.com/user-attachments/assets/6f4de25a-9ddd-4b03-8cdf-8985a0677cbf" />


---

## Performance Metrics (Example)

- Financial Stress Prediction Accuracy: 90%+  
- Anomaly Detection Precision: 88%  
- High-Risk Classification Accuracy: 92%  

*(Actual results depend on dataset and training configuration)*

---

## Results and Impact

The system demonstrates how Artificial Intelligence can transform traditional financial monitoring into a proactive risk intelligence framework.

By combining transaction flow analysis with predictive modeling, organizations can:

- Prevent financial collapse  
- Improve cash flow planning  
- Enhance risk governance  
- Support MSME sustainability  
- Enable data-driven executive decisions  

The project highlights how AI-powered financial analytics can provide scalable, real-time risk assessment solutions for modern enterprises.

---

## Installation Guide

### 1. Clone Repository

```bash
git clone https://github.com/your-username/AI-Driven-Financial-Stress-Risk-and-Survival-Analytics-System-Using-Transaction-Amount-Flow.git
cd AI-Driven-Financial-Stress-Risk-and-Survival-Analytics-System-Using-Transaction-Amount-Flow
