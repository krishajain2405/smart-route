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

# --- 1. PAGE CONFIGURATION (Must be at the top) ---
st.set_page_config(page_title="Smart Bin Analytics", layout="wide")

# --- 2. LOAD DATA (Your exact logic) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # Standardize columns for routing page
        rename_dict = {
            'bin_location_lat': 'lat', 'bin_location_lon': 'lon',
            'bin_fill_percent': 'fill', 'bin_id': 'bin_id'
        }
        for old, new in rename_dict.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization", "Impact & Financial Analysis"])

# --- 4. NAVIGATION LOGIC (Keeping every single part of your code) ---

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
    if df is not None:
        st.subheader("Dataset at a Glance")
        st.dataframe(df.head())
        st.write(f"The dataset contains *{len(df)}* hourly readings from *{df['bin_id'].nunique()}* simulated smart bins.")

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("These charts are now interactive. You can zoom, pan, and hover over the data.")

    st.subheader("Average Bin Fill Percentage by Hour of Day")
    # Using 'fill' because we renamed it in load_data for consistency
    hourly_fill_pattern = df.groupby(['hour_of_day', 'area_type'])['fill'].mean().reset_index()
    fig1 = px.line(hourly_fill_pattern, x='hour_of_day', y='fill', color='area_type',
                   title='Average Bin Fill Percentage by Hour of Day',
                   labels={'hour_of_day': 'Hour of Day', 'fill': 'Average Fill Level (%)'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Average Bin Fill Percentage by Day of the Week")
    daily_avg = df.groupby('day_of_week')['fill'].mean().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig2 = px.bar(daily_avg, x='day_of_week', y='fill', category_orders={"day_of_week": day_order},
                  title='Average Bin Fill Percentage by Day of the Week',
                  labels={'day_of_week': 'Day of the Week', 'fill': 'Average Fill Level (%)'})
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Predictive Model":
    st.title("Predictive Model for Bin Fill Level")
    with st.spinner("Preparing data and training model..."):
        features_to_use = ['hour_of_day', 'day_of_week', 'ward', 'area_type', 'time_since_last_pickup']
        target_variable = 'fill'
        model_df = df[features_to_use + [target_variable]].copy()
        for feature in ['day_of_week', 'ward', 'area_type']:
            if feature in model_df.columns:
                model_df = pd.get_dummies(model_df, columns=[feature], prefix=feature, drop_first=True)
        X = model_df.drop(target_variable, axis=1, errors='ignore')
        y = model_df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}%")
    col2.metric("R-squared (R²) Score", f"{r2:.2f}")

    plot_data = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    plot_data['Error'] = abs(plot_data['Actual'] - plot_data['Predicted'])
    fig = px.scatter(plot_data.sample(min(5000, len(plot_data))), x='Actual', y='Predicted', color='Error',
                     title="Actual vs. Predicted Fill Levels", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Route Optimization":
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
    
    # Map logic using YOUR lat/lon columns
    m = folium.Map(location=[19.0760, 72.8777], zoom_start=12, tiles="CartoDB positron")
    for _, row in df.iterrows():
        color = 'red' if row['fill'] >= threshold else 'green'
        folium.Marker([row['lat'], row['lon']], 
                      icon=folium.Icon(color=color, icon='trash', prefix='fa')).add_to(m)
    st_folium(m, width=1200, height=550)

    st.subheader(f"📲 Driver QR: Trip Ready")
    qr = qrcode.make(f"http://maps.google.com/maps?q={DEONAR_DUMPING[0]},{DEONAR_DUMPING[1]}")
    buf = BytesIO(); qr.save(buf)
    st.image(buf, width=200)

elif page == "Impact & Financial Analysis":
    st.title("💎 Comprehensive Business & Impact Model")
    st.write("This advanced model evaluates the project's viability.")

    st.sidebar.header("⚙ Simulation Parameters")
    is_ev = st.sidebar.checkbox("⚡ Activate Electric Vehicle (EV) Fleet Mode")

    # --- YOUR FULL FINANCIAL LOGIC (UNCUT) ---
    st.sidebar.subheader("1. CAPEX (Initial Investment)")
    num_trucks = st.sidebar.number_input("Fleet Size (Trucks)", value=5, min_value=1)
    hardware_cost_per_bin = st.sidebar.number_input("Hardware Cost/Bin (₹)", value=1500)
    total_bins = st.sidebar.number_input("Total Smart Bins", value=100)
    software_dev_cost = st.sidebar.number_input("Software/Cloud Setup Cost (₹)", value=50000)

    st.sidebar.subheader("2. OPEX (Operational)")
    driver_wage = st.sidebar.slider("Staff Hourly Wage (₹)", 100, 500, 200)

    if is_ev:
        fuel_price = st.sidebar.number_input("Electricity Cost (₹/kWh)", value=10.0)
        truck_efficiency = st.sidebar.number_input("EV Efficiency (km/kWh)", value=1.5)
        co2_factor = 0.82
    else:
        fuel_price = st.sidebar.number_input("Diesel Price (₹/Liter)", value=104.0)
        truck_efficiency = st.sidebar.number_input("Truck Mileage (km/L)", value=4.0)
        co2_factor = 2.68

    maintenance_per_km = st.sidebar.number_input("Vehicle Maint. (₹/km)", value=5.0)
    cloud_cost_per_bin = st.sidebar.number_input("Cloud/Data Cost per Bin/Month (₹)", value=20)

    st.sidebar.subheader("3. Revenue & Strategic Value")
    recyclable_value_per_kg = st.sidebar.number_input("Avg. Recyclable Value (₹/kg)", value=15.0)
    recycling_rate_increase = st.sidebar.slider("Recycling Efficiency Boost (%)", 0, 50, 20)
    daily_waste_collected_kg = st.sidebar.number_input("Total Daily Waste (kg)", value=2000.0)
    penalty_per_overflow = st.sidebar.number_input("Fine per Overflowing Bin (₹)", value=500)
    overflows_prevented_month = st.sidebar.slider("Overflows Prevented/Month", 0, 100, 25)
    carbon_credit_price = st.sidebar.number_input("Carbon Credit Price (₹/Ton CO2)", value=1500.0)

    st.sidebar.subheader("4. Logistics Efficiency")
    dist_old = st.sidebar.number_input("Daily Dist. Fixed (km)", value=60.0)
    trips_old = st.sidebar.slider("Trips/Month (Fixed)", 15, 30, 30)
    hours_old = st.sidebar.number_input("Hours/Trip (Fixed)", value=7.0)
    dist_new = st.sidebar.number_input("Daily Dist. Smart (km)", value=40.0)
    trips_new = st.sidebar.slider("Trips/Month (Smart)", 15, 30, 24)
    hours_new = st.sidebar.number_input("Hours/Trip (Smart)", value=5.0)

    # --- YOUR CALCULATIONS ---
    total_capex = (hardware_cost_per_bin * total_bins) + software_dev_cost
    def calc_opex(dist, trips, hours):
        total_dist = dist * trips * num_trucks
        total_hours = hours * trips * num_trucks
        energy_consumed = total_dist / truck_efficiency
        energy_cost = energy_consumed * fuel_price
        labor_cost = total_hours * driver_wage
        maint_cost = total_dist * maintenance_per_km
        co2_kg = energy_consumed * co2_factor
        return {"total_opex": energy_cost + labor_cost + maint_cost, "co2": co2_kg}

    old = calc_opex(dist_old, trips_old, hours_old)
    new = calc_opex(dist_new, trips_new, hours_new)
    new["total_opex"] += (cloud_cost_per_bin * total_bins)

    opex_savings = old["total_opex"] - new["total_opex"]
    revenue_gain = daily_waste_collected_kg * 30 * recyclable_value_per_kg * (recycling_rate_increase/100)
    penalty_savings = overflows_prevented_month * penalty_per_overflow
    co2_saved_tons = (old["co2"] - new["co2"]) / 1000
    carbon_credit_revenue = co2_saved_tons * carbon_credit_price
    total_monthly_benefit = opex_savings + revenue_gain + penalty_savings + carbon_credit_revenue
    months_breakeven = total_capex / total_monthly_benefit if total_monthly_benefit > 0 else 0

    st.markdown("### 📊 Monthly Financial Snapshot")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net Monthly Benefit", f"₹{total_monthly_benefit:,.0f}")
    k2.metric("Direct OPEX Savings", f"₹{opex_savings:,.0f}")
    k3.metric("Revenue Gain", f"₹{revenue_gain:,.0f}")
    k4.metric("ROI Break-even", f"{months_breakeven:.1f} Mo")

    fig_water = px.bar(x=["Savings", "Revenue", "Penalties", "Carbon"], 
                       y=[opex_savings, revenue_gain, penalty_savings, carbon_credit_revenue],
                       title="Value Breakdown")
    st.plotly_chart(fig_water, use_container_width=True)
    
    st.success(f"Final Verdict: Project break-even in {months_breakeven:.1f} months.")
