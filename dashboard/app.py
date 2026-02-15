import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from lifelines import KaplanMeierFitter

st.set_page_config(
    page_title="Customer Churn Intelligence",
    layout="wide"
)

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/featured.csv")

@st.cache_resource
def load_model():
    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


df = load_data()
model, scaler = load_model()

# Encode like training
df_encoded = pd.get_dummies(
    df,
    columns=["country", "subscription_plan"],
    drop_first=True
)

X = df_encoded.drop(["customer_id", "churn"], axis=1)
X_scaled = scaler.transform(X)

df["risk_score"] = model.predict_proba(X_scaled)[:,1]

df["risk_segment"] = pd.cut(
    df["risk_score"],
    bins=[-1,0.4,0.7,1],
    labels=["LOW","MEDIUM","HIGH"]
)

# -----------------------------
# TITLE
# -----------------------------

st.title("📊 Customer Churn Intelligence Dashboard")

st.markdown("AI-powered analytics to predict churn and protect recurring revenue.")

# -----------------------------
# EXECUTIVE KPIs
# -----------------------------

total_customers = len(df)
churn_rate = df["churn"].mean()*100
high_risk = len(df[df["risk_segment"]=="HIGH"])
revenue_at_risk = df[df["risk_segment"]=="HIGH"]["monthly_fee"].sum()*12

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churn Rate", f"{churn_rate:.2f}%")
col3.metric("High Risk Customers", high_risk)
col4.metric("Revenue At Risk", f"${revenue_at_risk:,.0f}")

st.divider()

# -----------------------------
# CHURN DISTRIBUTION
# -----------------------------

col1, col2 = st.columns(2)

fig = px.histogram(
    df,
    x="risk_score",
    nbins=50,
    title="Churn Risk Distribution"
)

col1.plotly_chart(fig, use_container_width=True)

segment_chart = px.pie(
    df,
    names="risk_segment",
    title="Customer Risk Segments"
)

col2.plotly_chart(segment_chart, use_container_width=True)

st.divider()

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

st.subheader("🔎 Key Drivers of Churn")

try:

    importances = model.feature_importances_
    features = X.columns

    importance_df = pd.DataFrame({
        "feature":features,
        "importance":importances
    }).sort_values("importance",ascending=False).head(10)

    fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Features Driving Churn"
    )

    st.plotly_chart(fig, use_container_width=True)

except:
    st.info("Feature importance unavailable for this model.")

st.divider()

# -----------------------------
# CUSTOMER LOOKUP
# -----------------------------

st.subheader("🔍 Customer Risk Lookup")

customer_id = st.selectbox(
    "Select Customer ID",
    df["customer_id"]
)

customer = df[df["customer_id"]==customer_id]

risk = float(customer["risk_score"])
segment = customer["risk_segment"].values[0]

st.write("### Risk Score:", round(risk,2))
st.write("### Segment:", segment)

if segment=="HIGH":
    st.error("Immediate retention action recommended!")

elif segment=="MEDIUM":
    st.warning("Monitor engagement closely.")

else:
    st.success("Customer likely to retain.")

st.divider()

# -----------------------------
# SURVIVAL CURVE
# -----------------------------

st.subheader("📉 Customer Survival Analysis")

kmf = KaplanMeierFitter()

T = df["tenure_months"]
E = df["churn"]

kmf.fit(T, event_observed=E)

survival_df = kmf.survival_function_.reset_index()

fig = px.line(
    survival_df,
    x="timeline",
    y="KM_estimate",
    title="Probability Customers Stay Over Time"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# SEGMENT ANALYTICS
# -----------------------------

st.subheader("📊 Segment-Level Churn")

segment_churn = df.groupby("subscription_plan")["churn"].mean().reset_index()

fig = px.bar(
    segment_churn,
    x="subscription_plan",
    y="churn",
    title="Churn Rate by Plan"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# SHAP explainability
# -----------------------------


st.subheader("🧠 Model Explainability (SHAP)")

st.write("Understand what factors are driving churn predictions.")

try:
    st.image("models/shap_summary.png")

except:
    st.info("Run explainability.py to generate SHAP plots.")
