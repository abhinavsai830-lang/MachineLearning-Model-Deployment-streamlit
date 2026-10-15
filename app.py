import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #edf2ff 0%, #dce6ff 50%, #eeeaff 100%); color: #172554; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #172554 0%, #3730a3 100%); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: #ffffff !important; }
[data-testid="stSidebar"] input { color: #111827 !important; background: #ffffff !important; }
h1, h2, h3, h4 { color: #172554 !important; font-weight: 750 !important; }
.hero { padding: 28px 32px; border-radius: 18px; background: linear-gradient(120deg, #f4b86a, #f59e55); margin-bottom: 20px; box-shadow: 0 8px 24px rgba(30,41,59,.12); }
.hero h1 { color: #ffffff !important; font-size: 2.2rem; margin-bottom: 8px; }
.hero p { color: #ffffff !important; margin-bottom: 0; }
.prediction-card { background: linear-gradient(120deg,#3730a3,#2563eb); padding:28px 20px; border-radius:18px; color:#ffffff !important; text-align:center; box-shadow:0 8px 20px rgba(37,99,235,.18); }
.prediction-card h3 { color:#ffffff !important; margin-bottom:16px; }
.prediction-price { color:#ffffff !important; font-size:2.4rem; font-weight:800; margin:12px 0; }
.prediction-card p { color:#e0e7ff !important; }
[data-testid="stMetric"] { background:#ffffff !important; padding:18px !important; border-radius:14px !important; border:1px solid #c7d2fe !important; box-shadow:0 4px 12px rgba(30,41,59,.06); }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color:#475569 !important; opacity:1 !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div, [data-testid="stMetricValue"] span { color:#172554 !important; opacity:1 !important; font-weight:750 !important; }
.stNumberInput label { color:#1e293b !important; font-weight:600 !important; }
.stNumberInput input { color:#111827 !important; background:#ffffff !important; }
div.stButton > button { background:#3730a3 !important; color:#ffffff !important; border:none !important; border-radius:10px !important; font-weight:700 !important; }
[data-testid="stCaptionContainer"] { color:#475569 !important; }
</style>
""", unsafe_allow_html=True)

RANDOM_SEED = 42
N_RECORDS = 1000
AGE_DEPRECIATION_LAKHS_PER_YEAR = 0.55

@st.cache_data
def generate_synthetic_data(n_rows=N_RECORDS, random_seed=RANDOM_SEED):
    """Generate artificial examples from an illustrative rule; no external data."""
    rng = np.random.default_rng(random_seed)
    area = rng.integers(500, 4001, size=n_rows)
    bedrooms = np.clip(np.rint(area / 650 + rng.normal(0, 0.65, n_rows)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, size=n_rows), 1, 5).astype(int)
    age = rng.integers(0, 61, size=n_rows)
    parking = rng.integers(0, 4, size=n_rows)
    base_price = 12.0 + 0.028 * area + 4.2 * bedrooms + 2.3 * bathrooms + 3.0 * parking
    noise = rng.normal(0, 2.5, size=n_rows)
    sale_price = np.maximum(0.0, base_price - AGE_DEPRECIATION_LAKHS_PER_YEAR * age + noise)
    data = pd.DataFrame({
        "Area_sqft": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
        "Age_years": age, "Parking": parking, "Price_lakhs": np.round(sale_price, 2)
    })
    return data, pd.Series(base_price, name="BasePrice_lakhs")

df, base_price_target = generate_synthetic_data()
features = ["Area_sqft", "Bedrooms", "Bathrooms", "Age_years", "Parking"]
non_age_features = ["Area_sqft", "Bedrooms", "Bathrooms", "Parking"]
X = df[features]
y = df["Price_lakhs"]

X_train, X_test, y_train, y_test, base_train, base_test = train_test_split(
    X, y, base_price_target, test_size=0.20, random_state=RANDOM_SEED
)
pricing_model = LinearRegression()
pricing_model.fit(X_train[non_age_features], base_train)

# Evaluate the same base-price + age-adjustment formula used by the live estimate.
base_test_predictions = pricing_model.predict(X_test[non_age_features])
y_pred = np.maximum(0.0, base_test_predictions - AGE_DEPRECIATION_LAKHS_PER_YEAR * X_test["Age_years"].to_numpy())
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = r2_score(y_test, y_pred)

st.markdown("""
<div class="hero"><h1>🏠 House Price Prediction</h1>
<p>Multiple Linear Regression | Synthetic data, model evaluation and house-price estimates</p></div>
""", unsafe_allow_html=True)
st.caption("Machine Learning Dashboard | Target: Price in ₹ lakhs")
st.warning("Synthetic-data demonstration only: this app generates 1,000 artificial records from an illustrative pricing rule. These metrics show fit to that rule, not real-world house-price accuracy. No external dataset is used.")

st.sidebar.title("🏡 Predict House Price")
st.sidebar.write("Enter the house details.")
area = st.sidebar.number_input("Area (square feet)", min_value=300, max_value=10000, value=2100, step=100)
bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=3, step=1)
age = st.sidebar.number_input("House Age (years)", min_value=0, max_value=100, value=5, step=1)
parking = st.sidebar.number_input("Parking Spaces", min_value=0, max_value=10, value=2, step=1)

input_data = pd.DataFrame([{
    "Area_sqft": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
    "Age_years": age, "Parking": parking
}])
base_price = float(pricing_model.predict(input_data[non_age_features])[0])
predicted_price = max(0.0, base_price - AGE_DEPRECIATION_LAKHS_PER_YEAR * age)
st.sidebar.divider()
st.sidebar.subheader("Selected Features")
st.sidebar.dataframe(input_data, hide_index=True, use_container_width=True)

st.markdown("## 🎯 House Price Prediction")
col1, col2 = st.columns([1.5, 1], gap="large")
with col1:
    st.markdown(f"""
    <div class="prediction-card"><h3>Estimated Demonstration Price</h3>
    <div class="prediction-price">₹ {predicted_price:.2f} Lakhs</div>
    <p>Base price adjusted for the selected house age</p></div>
    """, unsafe_allow_html=True)
with col2:
    st.metric("Input Features", len(features))
    st.metric("Synthetic Records", len(df))
    st.metric("Training Records", len(X_train))
    st.metric("Testing Records", len(X_test))
st.info(f"Age is handled explicitly: ₹{AGE_DEPRECIATION_LAKHS_PER_YEAR:.2f} lakhs are subtracted per additional year, holding other inputs fixed. This is an illustrative assumption, not a verified market rate.")

st.markdown("## 📊 Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", f"{mae:.2f} Lakhs")
m2.metric("MSE", f"{mse:.2f}")
m3.metric("RMSE", f"{rmse:.2f} Lakhs")
m4.metric("R² Score", f"{r2:.3f}")
st.caption("Metrics are evaluated on held-out synthetic records using the same pricing formula as the live estimate.")

st.markdown("## 🧾 Synthetic Dataset Preview")
with st.expander("View synthetic dataset records", expanded=True):
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.write(f"Dataset size: {df.shape[0]} rows × {df.shape[1]} columns")

st.markdown("## 📈 Visualizations")
tab1, tab2, tab3 = st.tabs(["Area vs Price", "Actual vs Predicted", "Feature Coefficients"])
with tab1:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df["Area_sqft"], df["Price_lakhs"], alpha=0.55, s=20, label="Synthetic records")
    slope, intercept = np.polyfit(df["Area_sqft"], df["Price_lakhs"], 1)
    x_values = np.linspace(df["Area_sqft"].min(), df["Area_sqft"].max(), 100)
    ax.plot(x_values, slope * x_values + intercept, "--", linewidth=2, label="Area-only trend")
    ax.set_title("House Area vs Synthetic Price", color="#172554")
    ax.set_xlabel("Area (square feet)")
    ax.set_ylabel("Price (₹ lakhs)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    comparison = pd.DataFrame({"Actual Price": y_test.to_numpy(), "Predicted Price": y_pred})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(comparison["Actual Price"], comparison["Predicted Price"], s=32, alpha=0.7)
    low = min(comparison["Actual Price"].min(), comparison["Predicted Price"].min())
    high = max(comparison["Actual Price"].max(), comparison["Predicted Price"].max())
    ax.plot([low, high], [low, high], "--", linewidth=2, label="Perfect prediction")
    ax.set_title("Actual vs Predicted (Synthetic Test Set)", color="#172554")
    ax.set_xlabel("Actual Price (₹ lakhs)")
    ax.set_ylabel("Predicted Price (₹ lakhs)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(comparison.round(2), hide_index=True, use_container_width=True)

with tab3:
    coefficient_df = pd.DataFrame({"Feature": non_age_features, "Coefficient": pricing_model.coef_}).sort_values("Coefficient")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(coefficient_df["Feature"], coefficient_df["Coefficient"], color="#4f46e5")
    ax.axvline(0, linewidth=1, color="#334155")
    ax.set_title("Base-Price Model Coefficients (Age Excluded)", color="#172554")
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(coefficient_df.round(4), hide_index=True, use_container_width=True)
    st.caption("The base-price model excludes age; the explicit age depreciation rule is applied separately.")

st.markdown("## 🧮 Prediction Equation")
equation_parts = [f"({coef:.4f} × {feature})" for feature, coef in zip(non_age_features, pricing_model.coef_)]
equation = f"Base Price = {pricing_model.intercept_:.4f} + " + " + ".join(equation_parts)
equation += f"\nPredicted Price = max(0, Base Price - {AGE_DEPRECIATION_LAKHS_PER_YEAR:.2f} × Age_years)"
st.code(equation, language="text")
with st.expander("Understand the metrics"):
    st.markdown("""
    - **MAE:** Average absolute prediction error, in lakhs.
    - **MSE:** Average squared prediction error.
    - **RMSE:** Square root of MSE, in lakhs.
    - **R²:** Fraction of target variation explained on the held-out synthetic set.
    
    These metrics are based on artificial data and are not a real-world valuation accuracy estimate.
    """)

st.divider()
st.caption("Built with Streamlit, pandas, NumPy, Matplotlib and scikit-learn. No external dataset is used.")
