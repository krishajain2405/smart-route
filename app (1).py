import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import qrcode
from io import BytesIO
import os

# --- Page Configuration ---
st.set_page_config(page_title="Smart Bin Analytics", layout="wide")

# --- Load and Cache Data ---
@st.cache_data
def load_data():
    # Detects your file automatically
    target = 'data.csv'
    if not os.path.exists(target):
        all_csvs = [f for f in os.listdir('.') if f.endswith('.csv')]
        target = all_csvs[0] if all_csvs else 'smart_bin_historical_data.csv'
    
    try:
        df = pd.read_csv(target, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower()
        # Mapping common names to standard ones
        df = df.rename(columns={
            'bin_location_lat': 'lat', 'bin_location_lon': 'lon',
            'bin_fill_percent': 'fill', 'timestamp': 'timestamp'
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        return df.dropna(subset=['timestamp'])
    except Exception as e:
        st.error(f"Load Error: {e}")
        return None

df = load_data()

# --- Multi-Truck Locations ---
GARAGES = {
    "Truck 1 (Worli)": (19.0178, 72.8478),
    "Truck 2 (Bandra)": (19.0596, 72.8295),
    "Truck 3 (Andheri)": (19.1136, 72.8697),
    "Truck 4 (Kurla)": (19.0726, 72.8844),
    "Truck 5 (Borivali)": (19.2307, 72.8567)
}
DEONAR_DUMPING = (19.0550, 72.9250)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization"])

# --- Home Page ---
if page == "Home":
    st.title("Smart Waste Management Analytics Dashboard")
    st.write("Welcome! This dashboard provides a comprehensive overview of the Smart Bin project's data science components.")
    st.subheader("Project Overview")
    st.write("""
    - **Live Data:** Real-time fill-level data sent to a cloud dashboard.
    - **Historical Analysis:** Patterns analyzed to understand waste generation.
    - **Predictive Modeling:** ML models forecasting bin overflow.
    - **Multi-Trip Optimization:** AI slicing massive workloads into manageable driver trips.
    """)
    st.subheader("Dataset at a Glance")
    st.dataframe(df.head())
    st.write(f"The dataset contains **{len(df)}** hourly readings.")

# --- EDA Page ---
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    
    st.subheader("Average Bin Fill Percentage by Hour of Day")
    hourly_fill = df.groupby(['hour_of_day', 'area_type'])['fill'].mean().reset_index()
    fig1 = px.line(hourly_fill, x='hour_of_day', y='fill', color='area_type', title='Waste Generation Cycle')
    st.plotly_chart(fig1, use_container_width=True)

# --- Predictive Model Page ---
elif page == "Predictive Model":
    st.title("Predictive Model for Bin Fill Level")
    
    with st.spinner("Training Random Forest model..."):
        features = ['hour_of_day', 'day_of_week', 'ward', 'area_type', 'time_since_last_pickup']
        target = 'fill'
        model_df = pd.get_dummies(df[features + [target]].dropna(), columns=['day_of_week', 'ward', 'area_type'], drop_first=True)
        X = model_df.drop(target, axis=1)
        y = model_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    st.subheader("Model Performance")
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:.2f}%")
    col2.metric("R² Score", f"{r2:.2f}")

# --- Route Optimization Page (Multi-Trip Pipeline) ---
elif page == "Route Optimization":
    st.title("🚛 AI Multi-Fleet Mission Control")

    # Sidebar Controls
    st.sidebar.header("🕹️ Dispatch Controls")
    selected_truck = st.sidebar.selectbox("Active Truck", list(GARAGES.keys()))
    threshold = st.sidebar.slider("Fill Threshold (%)", 0, 100, 75)
    
    # Simulation Time Slider (Fixes the Threshold problem)
    times = sorted(df['timestamp'].unique())
    sim_time = st.sidebar.select_slider("Select Simulation Time", options=times, value=times[int(len(times)*0.85)])
    
    df_snap = df[df['timestamp'] == sim_time].copy()

    # Assignment logic: Find nearest garage
    def assign_truck(row):
        loc = (row['lat'], row['lon'])
        dists = {name: ((loc[0]-c[0])**2 + (loc[1]-c[1])**2)**0.5 for name, c in GARAGES.items()}
        return min(dists, key=dists.get)

    df_snap['assigned_truck'] = df_snap.apply(assign_truck, axis=1)
    all_my_bins = df_snap[(df_snap['assigned_truck'] == selected_truck) & (df_snap['fill'] >= threshold)]
    all_my_bins = all_my_bins.sort_values('fill', ascending=False)

    # Multi-Trip Pipeline Logic
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Trip Pipeline")
    bins_per_trip = 8  # Keep QR code stable
    num_trips = (len(all_my_bins) // bins_per_trip) + (1 if len(all_my_bins) % bins_per_trip > 0 else 0)
    
    if num_trips > 0:
        trip_num = st.sidebar.selectbox(f"Select Trip (Total: {num_trips})", range(1, num
