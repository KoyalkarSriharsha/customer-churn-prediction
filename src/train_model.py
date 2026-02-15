import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def train():

    df = pd.read_csv("data/processed/featured.csv")

    df = pd.get_dummies(df, columns=["country", "subscription_plan"], drop_first=True)

    X = df.drop(["customer_id", "churn"], axis=1)
    y = df["churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "models/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    models = {

        "logistic": LogisticRegression(max_iter=1000),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6
        )
    }

    best_model = None
    best_auc = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)

        print(f"{name} AUC: {auc:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    joblib.dump(best_model, "models/churn_model.pkl")

    print("\nBest model saved!")
    print("Best AUC:", best_auc)


if __name__ == "__main__":
    train()
