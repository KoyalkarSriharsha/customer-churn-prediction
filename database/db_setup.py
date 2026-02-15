import sqlite3

def create_connection():
    conn = sqlite3.connect("database/churn.db")
    return conn


def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        age INTEGER,
        country TEXT,
        subscription_plan TEXT,
        tenure_months INTEGER,
        monthly_fee REAL,
        total_spend REAL,
        avg_logins_per_week INTEGER,
        support_tickets INTEGER,
        payment_delays INTEGER,
        last_login_days INTEGER,
        churn INTEGER
    )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_table()
    print("Database + table created successfully!")
