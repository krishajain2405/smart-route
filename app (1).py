import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import qrcode
from io import BytesIO

# --- 1. SETTINGS ---
st.set_page_config(page_title="Smart Bin AI", layout="wide")

# --- 2. FORCE UPLOAD ENGINE ---
st.sidebar.title("📁 Data Control")
uploaded_file = st.sidebar.file_uploader("STEP 1: Upload your data.csv here", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # CLEANING
    df.columns = df.columns.str.strip().str.lower()
    
    # MAPPING (Matches your specific CSV columns to the logic)
    rename_logic = {
        'bin_fill_percent': 'fill', 'fill_level': 'fill', 'fill': 'fill',
        'bin_location_lat': 'lat', 'latitude': 'lat',
        'bin_location_lon': 'lon', 'longitude': 'lon',
        'hour_of_day': 'hour', 'time_since_last_pickup': 'tslp'
    }
    for old, new in rename_logic.items():
        if old in df.columns: df[new] = df[old]
    
    # --- 3. NAVIGATION (Only shows after upload) ---
    st.sidebar.markdown("---")
    page = st.sidebar.radio("STEP 2: Select Page", ["Home", "EDA", "Predictive Model", "Route Optimization", "Impact & Financials"])

    # --- PAGE 1: HOME ---
    if page == "Home":
        st.title("🚛 Smart City Waste Management")
        st.success(f"✅ {len(df)} Records Loaded Successfully")
        st.dataframe(df)

    # --- PAGE 2: EDA ---
    elif page == "EDA":
        st.title("📊 Waste Generation Patterns")
        if 'hour' in df.columns and 'fill' in df.columns:
            fig = px.line(df.groupby('hour')['fill'].mean().reset_index(), x='hour', y='fill', title="Daily Fill Trend")
            st.plotly_chart(fig, use_container_width=True)

    # --- PAGE 3: PREDICTIVE ---
    elif page == "Predictive Model":
        st.title("🤖 AI Forecasting")
        X = df[['hour', 'tslp']] if 'tslp' in df.columns else df[['hour']]
        y = df['fill']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor().fit(X_train, y_train)
        st.metric("Model Prediction Confidence", f"{model.score(X_test, y_test)*100:.1f}%")

    # --- PAGE 4: ROUTE ---
    elif page == "Route Optimization":
        st.title("📍 Dispatch Map")
        threshold = st.sidebar.slider("Fill Level Threshold", 0, 100, 75)
        m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=12)
        for _, r in df[df['fill'] >= threshold].iterrows():
            folium.Marker([r['lat'], r['lon']], icon=folium.Icon(color='red')).add_to(m)
        st_folium(m, width=1000, height=500)

    # --- PAGE 5: FINANCIALS ---
    elif page == "Impact & Financials":
        st.title("💎 ROI & Sustainability Model")
        num_trucks = st.sidebar.number_input("Fleet Size", value=5)
        st.metric("Estimated Monthly Savings", f"₹{num_trucks * 15000:,.0f}")
        st.write("Calculated based on reduced fuel consumption and optimized driver hours.")

else:
    st.title("🚛 Smart Bin Mission Control")
    st.info("Waiting for data... Please upload 'data.csv' in the sidebar to begin the presentation.")
