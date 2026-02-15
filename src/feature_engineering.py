import pandas as pd
import numpy as np


def create_features():

    df = pd.read_csv("data/raw/saas_churn.csv")

    # -------------------------
    # RFM FEATURES
    # -------------------------

    df["recency_score"] = 1 / (df["last_login_days"] + 1)

    df["frequency_score"] = (
        df["avg_logins_per_week"] * 0.6 +
        df["support_tickets"] * 0.2
    )

    df["monetary_score"] = df["total_spend"]


    # -------------------------
    # Engagement Score
    # -------------------------

    df["engagement_score"] = (
        df["avg_logins_per_week"] * 0.5
        + df["tenure_months"] * 0.3
        - df["payment_delays"] * 0.2
    )


    # -------------------------
    # Trend Feature
    # Simulating usage decline
    # -------------------------

    np.random.seed(42)

    df["usage_decline"] = np.random.choice(
        [0, 1],
        size=len(df),
        p=[0.7, 0.3]
    )


    # -------------------------
    # CLV Estimate
    # -------------------------

    df["estimated_clv"] = (
        df["monthly_fee"]
        * df["tenure_months"]
        * (1 - df["churn"] * 0.5)
    )


    df.to_csv("data/processed/featured.csv", index=False)

    print("Feature engineering complete!")


if __name__ == "__main__":
    create_features()
