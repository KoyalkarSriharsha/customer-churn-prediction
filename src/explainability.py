import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

def generate_shap():

    df = pd.read_csv("data/processed/featured.csv")

    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    df_encoded = pd.get_dummies(
        df,
        columns=["country", "subscription_plan"],
        drop_first=True
    )

    X = df_encoded.drop(["customer_id","churn"], axis=1)
    X_scaled = scaler.transform(X)

    # Use TreeExplainer (fast for RF/XGBoost)
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_scaled)

    # -----------------------------
    # GLOBAL IMPORTANCE
    # -----------------------------

    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.savefig("models/shap_summary.png", bbox_inches='tight')
    plt.close()

    print("SHAP summary saved!")

    # -----------------------------
    # INDIVIDUAL EXPLANATION
    # -----------------------------

    plt.figure()

    shap.force_plot(
        explainer.expected_value,
        shap_values[0,:],
        X.iloc[0,:],
        matplotlib=True,
        show=False
    )

    plt.savefig("models/shap_force_plot.png", bbox_inches='tight')
    plt.close()

    print("Force plot saved!")


if __name__ == "__main__":
    generate_shap()
