import random
import sqlite3
from datetime import datetime, timedelta

DB_NAME = "C:/Users/prave/OneDrive/Desktop/Project/metabase/finguard.db"

branches = ["Coimbatore HQ", "Hosur Unit", "Madurai Export"]
categories = ["Sales", "Salary", "Raw Material", "EB", "EMI", "GST", "Other"]
entities = [
    "Customer A", "Customer B", "Supplier X", "Supplier Y",
    "Bank", "Government", "Employee Payroll"
]

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY,
            txn_date TEXT,
            entity_type TEXT,
            branch TEXT,
            from_entity TEXT,
            to_entity TEXT,
            amount REAL,
            txn_type TEXT,
            category TEXT
        );
    """)
    conn.commit()
    conn.close()

def random_date():
    days_ago = random.randint(0, 60)
    return (datetime.now().date() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

def insert_random_transactions(n=10000):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    for _ in range(n):
        entity_type = random.choice(["Company", "Branch"])
        branch = "Company" if entity_type == "Company" else random.choice(branches)

        txn_type = random.choice(["Credit", "Debit"])
        amount = random.randint(10, 100_000)
        category = random.choice(categories)

        if txn_type == "Credit":
            from_e = random.choice(entities)
            to_e = branch
        else:
            from_e = branch
            to_e = random.choice(entities)

        c.execute("""
            INSERT INTO transactions (
                txn_date,
                entity_type,
                branch,
                from_entity,
                to_entity,
                amount,
                txn_type,
                category
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            random_date(),
            entity_type,
            branch,
            from_e,
            to_e,
            amount,
            txn_type,
            category
        ))

    conn.commit()
    conn.close()
    print(f"✅ Inserted {n} random transactions")

# ---------- RUN ----------
init_db()
insert_random_transactions()
