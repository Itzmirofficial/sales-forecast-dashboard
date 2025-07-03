import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="Advanced Sales Forecast Dashboard", layout="wide")
st.title("Advanced Store-Item Sales Forecasting with XGBoost")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", parse_dates=["date"])
    return df

# --- FEATURE ENGINEERING FUNCTION ---
def engineer_features(df):
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekday"] = df["date"].dt.weekday
    df["lag_1"] = df["sales"].shift(1)
    df["rolling_3"] = df["sales"].rolling(3).mean()
    df["rolling_7"] = df["sales"].rolling(7).mean()
    df.dropna(inplace=True)
    return df

# --- LOAD AND FILTER DATA ---
df = load_data()

st.sidebar.header("Filter Options")
store = st.sidebar.selectbox("Store", sorted(df["store"].unique()))
item = st.sidebar.selectbox("Item", sorted(df["item"].unique()))
forecast_days = st.sidebar.slider("Days to Forecast", 30, 180, 90)

filtered_df = df[(df["store"] == store) & (df["item"] == item)].copy()
filtered_df.sort_values("date", inplace=True)
filtered_df = engineer_features(filtered_df)

# --- TRAIN MODEL ---
X = filtered_df[["day", "month", "year", "weekday", "lag_1", "rolling_3", "rolling_7"]]
y = filtered_df["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=forecast_days)
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- METRICS ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- CHART: Actual vs Predicted ---
st.subheader(f"Forecast for Store {store}, Item {item}")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=filtered_df["date"].iloc[-len(y_test):], y=y_test, name="Actual Sales"))
fig1.add_trace(go.Scatter(x=filtered_df["date"].iloc[-len(y_test):], y=y_pred, name="Predicted Sales"))
fig1.update_layout(title="Actual vs Predicted Sales", xaxis_title="Date", yaxis_title="Sales")
st.plotly_chart(fig1, use_container_width=True)

# --- METRICS SUMMARY ---
st.subheader("Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("R² Score", f"{r2:.2f}")
col4.metric("MSE", f"{mse:.2f}")

# --- FEATURE IMPORTANCE ---
st.subheader("Feature Importance")
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
fig_imp = go.Figure([go.Bar(x=importance_df["Feature"], y=importance_df["Importance"])])
fig_imp.update_layout(title="Model Feature Importance", xaxis_title="Feature", yaxis_title="Importance")
st.plotly_chart(fig_imp, use_container_width=True)

# --- HISTOGRAM OF RESIDUALS ---
st.subheader("Residual Error Distribution")
residuals = y_test - y_pred
fig2 = go.Figure(data=[go.Histogram(x=residuals)])
fig2.update_layout(title="Residuals (Actual - Predicted)", xaxis_title="Residual", yaxis_title="Frequency")
st.plotly_chart(fig2, use_container_width=True)

# --- ERROR OVER TIME ---
st.subheader("Error Over Time")
fig_error = go.Figure()
fig_error.add_trace(go.Scatter(x=filtered_df["date"].iloc[-len(y_test):], y=residuals, name="Prediction Error"))
fig_error.update_layout(title="Prediction Error Over Time", xaxis_title="Date", yaxis_title="Residual")
st.plotly_chart(fig_error, use_container_width=True)

# --- TIME SERIES DECOMPOSITION VIEW ---
st.subheader("Sales Trends and Rolling Averages")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=filtered_df["date"], y=filtered_df["sales"], name="Sales"))
fig3.add_trace(go.Scatter(x=filtered_df["date"], y=filtered_df["rolling_3"], name="Rolling 3-Day Avg"))
fig3.add_trace(go.Scatter(x=filtered_df["date"], y=filtered_df["rolling_7"], name="Rolling 7-Day Avg"))
fig3.update_layout(title="Sales with Rolling Averages", xaxis_title="Date", yaxis_title="Sales")
st.plotly_chart(fig3, use_container_width=True)

# --- HEATMAP SALES BY MONTH & WEEKDAY ---
st.subheader("Heatmap of Sales by Month & Weekday")
heat_df = filtered_df.copy()
heat_map = heat_df.groupby(["month", "weekday"])["sales"].mean().unstack()
fig_heat, ax = plt.subplots()
sns.heatmap(heat_map, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
st.pyplot(fig_heat)

# --- OUTLIER DATES ---
st.subheader("High Error Dates")
outliers = filtered_df.iloc[-len(y_test):].copy()
outliers["residual"] = residuals
high_error = outliers[np.abs(outliers["residual"]) > 2 * residuals.std()]
st.dataframe(high_error[["date", "sales", "residual"]].rename(columns={"sales": "Actual Sales"}))

# --- DOWNLOAD FORECAST CSV ---
forecast_df = pd.DataFrame({
    "Date": filtered_df["date"].iloc[-len(y_test):].values,
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})
csv = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast as CSV",
    data=csv,
    file_name=f"store_{store}_item_{item}_forecast.csv",
    mime="text/csv"
)

# --- REPORT SUMMARY ---
st.subheader("Summary Report")
st.markdown(f"""
**Developer**: Mir Abdul Aziz Khan

**Store ID**: `{store}`

**Item ID**: `{item}`

**Forecasting Horizon**: `{forecast_days}` days

**Model Used**: XGBoost Regressor

**Summary**:
- Sales forecasting model trained on store-item pair with engineered features like lag, rolling mean, and calendar variables.
- RMSE indicates an average prediction error of ±{rmse:.2f} units.
- R² score shows the model explains {r2:.0%} of the variance in sales.
- High error dates are flagged for operational insight.
- Feature importance and residual analysis included.
- Time series heatmap reveals seasonal sales patterns.

Use this dashboard to explore and export custom forecasts interactively.
""")

# --- PROJECT OVERVIEW ---
st.markdown("""
### Project Overview

This advanced sales forecasting application was developed by Mir Abdul Aziz Khan as a demonstration of best practices in modern machine learning, time series feature engineering, and data product deployment. It leverages the following:

Machine Learning:
- Model: XGBoost Regressor – chosen for its high performance with structured tabular data.
- Features: Calendar variables (day, month, weekday), lag features, and rolling averages to capture temporal dynamics.
- Evaluation: RMSE, MAE, R² for assessing performance.

Data Visualizations:
- Interactive comparison between actual vs predicted values
- Residual analysis to check model bias
- Rolling averages and decomposition
- Heatmap by weekday/month for demand insight
- Outlier identification

App Stack:
- Frontend: Streamlit for UI
- Backend: Python & Pandas
- Visuals: Plotly & Seaborn

Real-World Use Case:
- Helps supply chain managers, retailers, and analysts make informed restocking decisions.
- Easily scalable for multiple products and stores.

This dashboard represents an end-to-end data science product, integrating engineering, analysis, modeling, UI, and interactivity into a single professional-grade tool.
""")
