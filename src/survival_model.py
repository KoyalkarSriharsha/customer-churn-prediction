import pandas as pd
from lifelines import KaplanMeierFitter


def survival_analysis():

    df = pd.read_csv("data/processed/featured.csv")

    kmf = KaplanMeierFitter()

    T = df["tenure_months"]
    E = df["churn"]

    kmf.fit(T, event_observed=E)

    kmf.plot_survival_function()

    print("Median survival time:", kmf.median_survival_time_)


if __name__ == "__main__":
    survival_analysis()
