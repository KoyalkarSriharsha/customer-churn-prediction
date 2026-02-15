import pandas as pd
import joblib


def evaluate_business_impact():

    df = pd.read_csv("data/processed/featured.csv")

    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    df_encoded = pd.get_dummies(
        df,
        columns=["country", "subscription_plan"],
        drop_first=True
    )

    X = df_encoded.drop(["customer_id", "churn"], axis=1)

    X_scaled = scaler.transform(X)

    df["risk_score"] = model.predict_proba(X_scaled)[:, 1]

    # SEGMENTATION
    conditions = [
        df["risk_score"] > 0.7,
        df["risk_score"].between(0.4, 0.7),
        df["risk_score"] < 0.4
    ]

    choices = ["HIGH", "MEDIUM", "LOW"]

    df["risk_segment"] = pd.Series(
        pd.cut(
            df["risk_score"],
            bins=[-1,0.4,0.7,1],
            labels=["LOW","MEDIUM","HIGH"]
        )
    )

    # Revenue at Risk
    revenue_at_risk = df[df["risk_segment"] == "HIGH"]["monthly_fee"].sum() * 12

    print("\n========= BUSINESS IMPACT =========")
    print("High Risk Customers:", len(df[df["risk_segment"]=="HIGH"]))
    print("Revenue At Risk (Annual): $", round(revenue_at_risk,2))


if __name__ == "__main__":
    evaluate_business_impact()
