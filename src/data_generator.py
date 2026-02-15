import pandas as pd
import numpy as np
from faker import Faker
import random
import sqlite3

fake = Faker()

PLANS = ["Basic", "Pro", "Enterprise"]
COUNTRIES = ["USA", "India", "UK", "Germany", "Canada"]

def generate_customer():
    tenure = np.random.randint(1, 48)

    avg_logins = np.random.poisson(5)
    tickets = np.random.poisson(2)
    delays = np.random.poisson(1)

    plan = random.choice(PLANS)

    fee_map = {
        "Basic": 20,
        "Pro": 50,
        "Enterprise": 120
    }

    monthly_fee = fee_map[plan]
    total_spend = monthly_fee * tenure

    last_login = np.random.randint(1, 60)

    # churn probability logic (IMPORTANT for realism)
    churn_prob = (
        0.3 * (avg_logins < 3) +
        0.2 * (tickets > 4) +
        0.3 * (delays > 2) +
        0.2 * (last_login > 30)
    )

    churn = np.random.binomial(1, min(churn_prob, 0.9))

    return [
        fake.uuid4(),
        np.random.randint(18, 70),
        random.choice(COUNTRIES),
        plan,
        tenure,
        monthly_fee,
        total_spend,
        avg_logins,
        tickets,
        delays,
        last_login,
        churn
    ]


def generate_dataset(n=10000):

    columns = [
        "customer_id","age","country","subscription_plan",
        "tenure_months","monthly_fee","total_spend",
        "avg_logins_per_week","support_tickets",
        "payment_delays","last_login_days","churn"
    ]

    data = [generate_customer() for _ in range(n)]

    df = pd.DataFrame(data, columns=columns)

    df.to_csv("data/raw/saas_churn.csv", index=False)

    print("Dataset created!")


def insert_into_db():

    conn = sqlite3.connect("database/churn.db")
    df = pd.read_csv("data/raw/saas_churn.csv")

    df.to_sql("customers", conn, if_exists="replace", index=False)

    conn.close()
    print("Data inserted into SQLite!")


if __name__ == "__main__":
    generate_dataset()
    insert_into_db()
