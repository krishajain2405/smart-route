import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import osmnx as ox
import networkx as nx
import qrcode
from io import BytesIO
import os

# --- Page Configuration (Called ONLY once) ---
st.set_page_config(page_title="Smart Bin Analytics", layout="wide")

# --- Load and Cache Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data.csv')
        # Standardize column names for the script
        df.columns = [c.strip().lower() for c in df.columns]
        rename_dict = {
            'bin_location_lat': 'lat', 'bin_location_lon': 'lon',
            'bin_fill_percent': 'fill', 'timestamp': 'timestamp',
            'bin_id': 'bin_id'
        }
        for old, new in rename_dict.items():
            if old in df.columns: df = df.rename(columns={old: new})
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except:
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2025-01-01 10:00:00']),
            'bin_id': ['B101'], 'hour_of_day': [10], 'day_of_week': ['Monday'],
            'ward': ['Ward_A'], 'area_type': ['Residential'], 'fill': [50],
            'lat': [19.0760], 'lon': [72.8777]
        })
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization", "Impact & Financial Analysis"])

# --- Home Page ---
if page == "Home":
    st.title("Smart Waste Management Analytics Dashboard")
    st.write("Welcome! This dashboard provides a comprehensive overview of the Smart Bin project's data science components.")
    st.subheader("Project Overview")
    st.write("""
    - *Live Data:* A physical prototype sends real-time fill-level data to a cloud dashboard.
    - *Historical Analysis:* We analyze a large dataset to understand waste generation patterns.
    - *Predictive Modeling:* A machine learning model forecasts when bins will become full.
    - *Route Optimization:* An algorithm calculates the most efficient collection route for full bins.
    - *Financial Impact:* A comprehensive model calculating ROI, carbon credits, and operational savings.
    """)
    st.subheader("Dataset at a Glance")
    st.dataframe(df.head())

# --- EDA Page ---
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    # ... (Your Plotly Chart Code Here) ...
    st.info("Visualizing historical patterns and fill trends.")

# --- Predictive Model Page ---
elif page == "Predictive Model":
    st.title("Predictive Model for Bin Fill Level")
    # ... (Your Random Forest Code Here) ...

# --- Route Optimization Page ---
elif page == "Route Optimization":
    # EVERYTHING related to Route Optimization stays inside this ELIF block
    st.title("🚛 AI Multi-Fleet Mission Control")
    
    GARAGES = {
        "Truck 1 (Worli)": (19.0178, 72.8478),
        "Truck 2 (Bandra)": (19.0596, 72.8295),
        "Truck 3 (Andheri)": (19.1136, 72.8697),
        "Truck 4 (Kurla)": (19.0726, 72.8844),
        "Truck 5 (Borivali)": (19.2307, 72.8567)
    }
    DEONAR_DUMPING = (19.0550, 72.9250)

    st.sidebar.header("🕹 Dispatch Controls")
    selected_truck = st.sidebar.selectbox("Select Active Truck", list(GARAGES.keys()))
    threshold = st.sidebar.slider("Fill Threshold (%)", 0, 100, 75)
    
    # Mission Control Map and QR Code logic
    try:
        m = folium.Map(location=[19.0760, 72.8777], zoom_start=12, tiles="CartoDB positron")
        # (Your Route Mapping/Folium code logic here)
        st_folium(m, width=1200, height=550)
        st.success("Mission route ready for dispatch.")
    except Exception as e:
        st.error(f"Mapping Error: {e}")

# --- Impact & Financial Analysis Page ---
elif page == "Impact & Financial Analysis":
    st.title("💎 Comprehensive Business & Impact Model")
    # ... (Your Financial Analysis sliders and Waterfall charts here) ...
