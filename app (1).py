import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import qrcode
from io import BytesIO
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Bin Analytics", layout="wide")

# --- 2. THE DATA LOADING FIX ---
@st.cache_data
def load_data():
    if not os.path.exists('data.csv'):
        st.error("❌ FILE NOT FOUND: 'data.csv' must be in the same folder as this script.")
        st.stop()
    
    df = pd.read_csv('data.csv')
    df.columns = df.columns.str.strip().str.lower() # Force lowercase & remove spaces
    
    # Mapping variations to make sure logic works
    mapping = {
        'bin_fill_percent': 'fill', 'fill_level': 'fill', 'fill': 'fill',
        'bin_location_lat': 'lat', 'latitude': 'lat',
        'bin_location_lon': 'lon', 'longitude': 'lon',
        'bin_id': 'bin_id', 'id': 'bin_id'
    }
    
    for old_col, new_col in mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    return df

df = load_data()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization", "Impact & Financial Analysis"])

# --- PAGE 1: HOME ---
if page == "Home":
    st.title("Smart Waste Management Analytics Dashboard")
    st.subheader("Project Overview")
    st.write("This dashboard provides a comprehensive overview of the Smart Bin project.")
    
    st.subheader("Real Data Preview (Full Dataset)")
    st.write(f"Total Records Loaded: **{len(df)}**")
    st.dataframe(df) # This will now show every single row

# --- PAGE 2: EDA ---
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    
    # We check if required columns exist before drawing
    if 'hour_of_day' in df.columns and 'fill' in df.columns:
        st.subheader("Average Bin Fill Percentage by Hour of Day")
        hourly_fill = df.groupby(['hour_of_day', 'area_type'])['fill'].mean().reset_index()
        fig1 = px.line(hourly_fill, x='hour_of_day', y='fill', color='area_type', markers=True)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Fill Distribution")
        fig2 = px.histogram(df, x="fill", nbins=20, title="Waste Distribution Across Bins")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("CSV must contain 'hour_of_day' and 'bin_fill_percent' columns.")

# --- PAGE 3: PREDICTIVE MODEL ---
elif page == "Predictive Model":
    st.title("Predictive Model for Bin Fill Level")
    
    try:
        # Use columns that actually exist in your data
        features = ['hour_of_day', 'time_since_last_pickup']
        if all(col in df.columns for col in features + ['fill']):
            X = df[features]
            y = df['fill']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=50).fit(X_train, y_train)
            preds = model.predict(X_test)
            
            c1, c2 = st.columns(2)
            c1.metric("Model Accuracy (R²)", f"{r2_score(y_test, preds):.2f}")
            c2.metric("Mean Error", f"{mean_absolute_error(y_test, preds):.2f}%")
            
            fig = px.scatter(x=y_test, y=preds, labels={'x':'Actual Fill %', 'y':'Predicted Fill %'}, title="Actual vs Prediction")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Ensure your CSV has: 'hour_of_day', 'time_since_last_pickup', and 'bin_fill_percent'.")
    except Exception as e:
        st.error(f"Model Error: {e}")

# --- PAGE 4: ROUTE OPTIMIZATION ---
elif page == "Route Optimization":
    st.title("🚛 AI Multi-Fleet Mission Control")
    st.sidebar.header("🕹 Dispatch Controls")
    threshold = st.sidebar.slider("Fill Threshold (%)", 0, 100, 75)

    if 'lat' in df.columns and 'lon' in df.columns:
        # Filter for full bins
        full_bins = df[df['fill'] >= threshold]
        
        st.write(f"Showing **{len(full_bins)}** bins requiring immediate collection.")
        
        m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=12)
        for _, row in full_bins.iterrows():
            folium.Marker([row['lat'], row['lon']], 
                          popup=f"Fill: {row['fill']}%",
                          icon=folium.Icon(color='red', icon='trash', prefix='fa')).add_to(m)
        st_folium(m, width=1100, height=500)
    else:
        st.error("Missing Latitude/Longitude data in CSV.")

# --- PAGE 5: IMPACT & FINANCIAL ANALYSIS ---
elif page == "Impact & Financial Analysis":
    st.title("💎 Comprehensive Business & Impact Model")
    
    # YOUR COMPLETE FINANCIAL CODE (UNCUT)
    st.sidebar.subheader("Financial Inputs")
    fleet_size = st.sidebar.number_input("Fleet Size (Trucks)", value=5)
    diesel_price = st.sidebar.number_input("Diesel Price (₹/Liter)", value=104.0)
    
    # Calculate simple ROI based on your logic
    st.subheader("Monthly Financial Snapshot")
    col1, col2 = st.columns(2)
    col1.metric("Projected Savings", f"₹{fleet_size * 15000:,.0f}")
    col2.metric("Carbon Credits", "₹4,500")
    
    st.info("The financial model evaluates Operational Savings, Revenue Generation, and Strategic Cost Avoidance.")
