import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

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

# =============================================================================
# DATA FUNCTIONS
# =============================================================================
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

def get_balance(entity_type, branch=None):
    conn = get_connection()
    if entity_type == "Company":
        query = """
            SELECT SUM(
                CASE WHEN txn_type='Credit' THEN amount ELSE -amount END
            ) FROM transactions WHERE entity_type='Company'
        """
        balance = pd.read_sql(query, conn).iloc[0,0]
    else:
        query = """
            SELECT SUM(
                CASE WHEN txn_type='Credit' THEN amount ELSE -amount END
            ) FROM transactions WHERE entity_type='Branch' AND branch=?
        """
        balance = pd.read_sql(query, conn, params=(branch,)).iloc[0,0]
    conn.close()
    return balance or 0

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

# =============================================================================
# POWER BI EMBED URLS
# =============================================================================
COMPANY_PBI_URL = "https://app.powerbi.com/view?r=YOUR_COMPANY_REPORT_ID"
BRANCH_PBI_URL = "https://app.powerbi.com/view?r=YOUR_BRANCH_REPORT_ID"

# =============================================================================
# DASHBOARD (COMPANY)
# =============================================================================
if st.session_state.page == "Dashboard":

    st.markdown("## 🏢 Kovai Precision Engineering")

    balance = get_balance("Company")
    st.metric("Company Net Cash", f"₹{balance:,.0f}")

    st.markdown("### 📊 Company Financial Dashboard (Power BI)")
    st.components.v1.iframe(
        COMPANY_PBI_URL,
        height=800,
        scrolling=True
    )

    st.markdown("### 🏭 Branches")
    for b in ["Coimbatore HQ", "Hosur Unit", "Madurai Export"]:
        if st.button(b):
            st.session_state.page = "Branch"
            st.session_state.branch = b
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
    st.markdown(f"## 🏭 {branch}")

    balance = get_balance("Branch", branch)
    st.metric("Branch Cash Balance", f"₹{balance:,.0f}")

    st.markdown("### 📊 Branch Analytics (Power BI)")
    st.components.v1.iframe(
        f"{BRANCH_PBI_URL}&filter=transactions/branch eq '{branch}'",
        height=800,
        scrolling=True
    )

    if st.button("⬅ Back"):
        st.session_state.page = "Dashboard"
        st.rerun()
