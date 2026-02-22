import os
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from datetime import timedelta

st.set_page_config(page_title="Retail Sales Forecast", layout="wide")

# -------------------------------------------------------
# DOWNLOAD DATA FROM KAGGLE
# -------------------------------------------------------
def download_data():
    if not os.path.exists("data/train.csv"):

        st.info("Downloading dataset from Kaggle...")

        # ✅ Check if secrets exist
        if "kaggle" not in st.secrets:
            st.error("Kaggle credentials not found in Streamlit secrets.")
            st.stop()

        os.makedirs("data", exist_ok=True)

        # Set environment variables
        os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
        os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            api.competition_download_files(
                "rossmann-store-sales",
                path="data"
            )

            # Extract ZIP
            zip_path = "data/rossmann-store-sales.zip"

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("data")

            os.remove(zip_path)

            st.success("Dataset downloaded successfully!")

        except Exception as e:
            st.error("Error downloading dataset. Make sure you joined the competition on Kaggle.")
            st.exception(e)
            st.stop()

download_data()

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("data/train.csv")
    store = pd.read_csv("data/store.csv")
    df = train.merge(store, on="Store", how="left")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# -------------------------------------------------------
# TRAIN MODEL IF NOT EXISTS
# -------------------------------------------------------
def train_model():

    if not os.path.exists("model/sales_model.pkl"):

        st.info("Training model... (first run may take 2-3 minutes)")

        os.makedirs("model", exist_ok=True)

        df_model = df.copy()

        # Date features
        df_model['Year'] = df_model['Date'].dt.year
        df_model['Month'] = df_model['Date'].dt.month
        df_model['Day'] = df_model['Date'].dt.day
        df_model['WeekOfYear'] = df_model['Date'].dt.isocalendar().week.astype(int)

        df_model = df_model.sort_values(["Store", "Date"])

        # Lag features
        df_model['Lag_1'] = df_model.groupby("Store")["Sales"].shift(1)
        df_model['Lag_7'] = df_model.groupby("Store")["Sales"].shift(7)
        df_model['Rolling_7'] = (
            df_model.groupby("Store")["Sales"]
            .shift(1)
            .rolling(7)
            .mean()
        )

        df_model = df_model[df_model["Open"] == 1]
        df_model = df_model.fillna(0)

        features = [
            'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday',
            'Year', 'Month', 'Day', 'WeekOfYear',
            'CompetitionDistance',
            'Lag_1', 'Lag_7', 'Rolling_7'
        ]

        X = df_model[features]
        y = df_model['Sales']

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

        joblib.dump(model, "model/sales_model.pkl")

        st.success("Model trained successfully!")

train_model()

model = joblib.load("model/sales_model.pkl")

# -------------------------------------------------------
# DASHBOARD UI
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

        future_dates = pd.date_range(start=start_date, periods=30)

        forecast_df = pd.DataFrame()
        forecast_df["Date"] = future_dates
        forecast_df["Store"] = store_id
        forecast_df["DayOfWeek"] = forecast_df["Date"].dt.dayofweek + 1
        forecast_df["Promo"] = 0
        forecast_df["SchoolHoliday"] = 0
        forecast_df["Year"] = forecast_df["Date"].dt.year
        forecast_df["Month"] = forecast_df["Date"].dt.month
        forecast_df["Day"] = forecast_df["Date"].dt.day
        forecast_df["WeekOfYear"] = forecast_df["Date"].dt.isocalendar().week.astype(int)

        competition = df[df["Store"] == store_id]["CompetitionDistance"].iloc[0]
        forecast_df["CompetitionDistance"] = competition

        last_sales = df[df["Store"] == store_id]["Sales"].iloc[-1]

        forecast_df["Lag_1"] = last_sales
        forecast_df["Lag_7"] = last_sales
        forecast_df["Rolling_7"] = last_sales

        features = [
            'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday',
            'Year', 'Month', 'Day', 'WeekOfYear',
            'CompetitionDistance',
            'Lag_1', 'Lag_7', 'Rolling_7'
        ]

        preds = model.predict(forecast_df[features])
        forecast_df["Predicted Sales"] = preds

        fig = px.line(
            forecast_df,
            x="Date",
            y="Predicted Sales",
            template="plotly_white",
            title="30-Day Sales Forecast"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast_df)
