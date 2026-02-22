import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

st.set_page_config(page_title="Retail Sales Forecast", layout="wide")

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("data/train.csv")
    store = pd.read_csv("data/store.csv")
    df = train.merge(store, on="Store", how="left")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# -------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------
def prepare_features(data):
    data = data.copy()

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)

    data = data.sort_values(["Store", "Date"])

    data["Lag_1"] = data.groupby("Store")["Sales"].shift(1)
    data["Lag_7"] = data.groupby("Store")["Sales"].shift(7)
    data["Rolling_7"] = (
        data.groupby("Store")["Sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    data = data[data["Open"] == 1]
    data = data.fillna(0)

    return data

df_model = prepare_features(df)

features = [
    'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday',
    'Year', 'Month', 'Day', 'WeekOfYear',
    'CompetitionDistance',
    'Lag_1', 'Lag_7', 'Rolling_7'
]

# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------
def train_model():
    if not os.path.exists("model/sales_model.pkl"):

        st.info("Training model... (First run may take 2-3 minutes)")

        os.makedirs("model", exist_ok=True)

        X = df_model[features]
        y = df_model["Sales"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        score = r2_score(y_test, preds)

        joblib.dump(model, "model/sales_model.pkl")
        st.success(f"Model trained! R² Score: {score:.3f}")

train_model()

model = joblib.load("model/sales_model.pkl")

# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.title("🏬 Retail Sales Intelligence Dashboard")

menu = st.sidebar.radio("Navigation", ["Executive Dashboard", "30-Day Forecast"])

# ---------------- DASHBOARD ----------------
if menu == "Executive Dashboard":

    total_sales = df["Sales"].sum()
    avg_sales = df["Sales"].mean()
    total_stores = df["Store"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Total Revenue", f"€ {int(total_sales):,}")
    col2.metric("📊 Avg Daily Sales", f"€ {int(avg_sales):,}")
    col3.metric("🏪 Active Stores", total_stores)

    sales_trend = df.groupby("Date")["Sales"].sum().reset_index()

    fig = px.line(
        sales_trend,
        x="Date",
        y="Sales",
        template="plotly_white",
        title="Daily Sales Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- FORECAST ----------------
if menu == "30-Day Forecast":

    store_id = st.selectbox("Select Store", sorted(df["Store"].unique()))
    start_date = st.date_input("Forecast Start Date")

    if st.button("Generate Forecast"):

        store_data = df[df["Store"] == store_id].sort_values("Date")
        last_7_days = list(store_data["Sales"].tail(7))

        future_dates = pd.date_range(start=start_date, periods=30)
        predictions = []

        for date in future_dates:

            lag_1 = last_7_days[-1]
            lag_7 = last_7_days[0]
            rolling_7 = np.mean(last_7_days)

            row = pd.DataFrame({
                "Store": [store_id],
                "DayOfWeek": [date.dayofweek + 1],
                "Promo": [0],
                "SchoolHoliday": [0],
                "Year": [date.year],
                "Month": [date.month],
                "Day": [date.day],
                "WeekOfYear": [date.isocalendar().week],
                "CompetitionDistance": [
                    store_data["CompetitionDistance"].iloc[0]
                ],
                "Lag_1": [lag_1],
                "Lag_7": [lag_7],
                "Rolling_7": [rolling_7]
            })

            pred = model.predict(row)[0]
            predictions.append(pred)

            last_7_days.append(pred)
            last_7_days.pop(0)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Sales": predictions
        })

        fig = px.line(
            forecast_df,
            x="Date",
            y="Predicted Sales",
            template="plotly_white",
            title="30-Day Rolling Sales Forecast"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast_df)
