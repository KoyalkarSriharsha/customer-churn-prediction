import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess():

    df = pd.read_csv("data/raw/saas_churn.csv")

    df = pd.get_dummies(df, columns=["country", "subscription_plan"], drop_first=True)

    X = df.drop(["customer_id", "churn"], axis=1)
    y = df["churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "models/scaler.pkl")

    processed = pd.DataFrame(X_scaled, columns=X.columns)
    processed["churn"] = y

    processed.to_csv("data/processed/processed.csv", index=False)

    print("Preprocessing complete!")


if __name__ == "__main__":
    preprocess()
