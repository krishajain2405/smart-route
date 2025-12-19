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
import osmnx as ox
import networkx as nx
import qrcode
from io import BytesIO
import os

# --- Page Configuration ---
st.set_page_config(page_title="Smart Bin Analytics", layout="wide")

# --- Multi-Fleet Hubs ---
GARAGES = {
    "Truck 1 (Worli)": (19.0178, 72.8478),
    "Truck 2 (Bandra)": (19.0596, 72.8295),
    "Truck 3 (Andheri)": (19.1136, 72.8697),
    "Truck 4 (Kurla)": (19.0726, 72.8844),
    "Truck 5 (Borivali)": (19.2307, 72.8567)
}
DEONAR_DUMPING = (19.0550, 72.9250)

# --- Load Data ---
@st.cache_data
def load_data():
    target = 'data.csv' if os.path.exists('data.csv') else ('smart_bin_historical_data.csv' if os.path.exists('smart_bin_historical_data.csv') else None)
    if not target: return None
    try:
        df = pd.read_csv(target, sep=None, engine='python', encoding='utf-8-sig')
        df.columns = [c.strip().lower() for c in df.columns]
        rename_dict = {'bin_location_lat': 'lat', 'bin_location_lon': 'lon', 'bin_fill_percent': 'fill', 'timestamp': 'timestamp'}
        for old, new in rename_dict.items():
            if old in df.columns: df = df.rename(columns={old: new})
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        return df.dropna(subset=['timestamp'])
    except: return None

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization", "Impact & Financial Analysis"])

# --- Home Page ---
if page == "Home":
    st.title("Smart Waste Management Analytics Dashboard")
    st.subheader("Project Overview")
    st.write("""
    - **Live Data:** Real-time physical prototype data.
    - **Historical Analysis:** Waste generation patterns.
    - **Predictive Modeling:** ML forecasts for bin overflows.
    - **Multi-Fleet Optimization:** AI-driven fleet routing and trip pipelines.
    - **Financial Impact:** ROI, carbon credits, and operational savings.
    """)
    st.dataframe(df.head())

# --- EDA Page ---
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    hourly_fill = df.groupby(['hour_of_day', 'area_type'])['fill'].mean().reset_index()
    fig1 = px.line(hourly_fill, x='hour_of_day', y='fill', color='area_type', title='Average Fill % by Hour')
    st.plotly_chart(fig1, width='stretch')

# --- Predictive Model Page ---
elif page == "Predictive Model":
    st.title("Predictive Model for Bin Fill Level")
    features = ['hour_of_day', 'day_of_week', 'ward', 'area_type', 'time_since_last_pickup']
    target = 'fill'
    model_df = pd.get_dummies(df[features + [target]].dropna(), drop_first=True)
    X, y = model_df.drop(target, axis=1), model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, n_jobs=-1).fit(X_train, y_train)
    preds = model.predict(X_test)
    st.metric("R² Confidence Score", f"{r2_score(y_test, preds):.2f}")
    fig3 = px.scatter(pd.DataFrame({'Actual': y_test, 'Predicted': preds}).sample(min(2000, len(y_test))), x='Actual', y='Predicted', trendline="ols")
    st.plotly_chart(fig3, width='stretch')

# --- Route Optimization ---
elif page == "Route Optimization":
    st.title("🚛 AI Multi-Fleet Route Optimization")
    selected_truck = st.sidebar.selectbox("Select Active Truck", list(GARAGES.keys()))
    threshold = st.sidebar.slider("Fill Threshold (%)", 0, 100, 75)
    times = sorted(df['timestamp'].unique())
    sim_time = st.sidebar.select_slider("Simulation Time", options=times, value=times[int(len(times)*0.85)])
    df_snap = df[df['timestamp'] == sim_time].copy()

    def assign_truck_logic(row):
        loc = (row['lat'], row['lon'])
        dists = {name: ((loc[0]-c[0])**2 + (loc[1]-c[1])**2)**0.5 for name, c in GARAGES.items()}
        return min(dists, key=dists.get)

    df_snap['assigned_truck'] = df_snap.apply(assign_truck_logic, axis=1)
    all_my_bins = df_snap[(df_snap['assigned_truck'] == selected_truck) & (df_snap['fill'] >= threshold)].sort_values('fill', ascending=False)
    
    st.sidebar.subheader("📦 Trip Pipeline")
    bins_per_trip = 8
    num_trips = (len(all_my_bins) // bins_per_trip) + (1 if len(all_my_bins) % bins_per_trip > 0 else 0)
    if num_trips > 0:
        trip_num = st.sidebar.selectbox(f"Select Trip (Total: {num_trips})", range(1, num_trips + 1), format_func=lambda x: f"Trip {x}")
        current_bins = all_my_bins.iloc[(trip_num-1)*bins_per_trip : trip_num*bins_per_trip]
    else: current_bins = pd.DataFrame()

    try:
        G = ox.graph_from_point((19.0760, 72.8777), dist=8000, network_type='drive')
        m = folium.Map(location=[19.0760, 72.8777], zoom_start=12, tiles="CartoDB positron")
        for _, row in df_snap.iterrows():
            is_full, is_mine = row['fill'] >= threshold, row['assigned_truck'] == selected_truck
            color = 'red' if is_full and is_mine else ('orange' if is_full else 'green')
            folium.Marker([row['lat'], row['lon']], icon=folium.Icon(color=color, icon='trash', prefix='fa')).add_to(m)

        garage_loc = GARAGES[selected_truck]
        if not current_bins.empty:
            pts = [garage_loc] + list(zip(current_bins['lat'], current_bins['lon'])) + [DEONAR_DUMPING]
            path_coords = []
            for i in range(len(pts)-1):
                n1, n2 = ox.nearest_nodes(G, pts[i][1], pts[i][0]), ox.nearest_nodes(G, pts[i+1][1], pts[i+1][0])
                route = nx.shortest_path(G, n1, n2, weight='length')
                path_coords.extend([[G.nodes[node]['y'], G.nodes[node]['x']] for node in route])
            folium.PolyLine(path_coords, color="#3498db", weight=6).add_to(m)
        folium.Marker(garage_loc, icon=folium.Icon(color='blue', icon='truck', prefix='fa')).add_to(m)
        folium.Marker(DEONAR_DUMPING, icon=folium.Icon(color='black', icon='home', prefix='fa')).add_to(m)
        st_folium(m, width=1200, height=550, key="mission_map")
        if not current_bins.empty:
            google_url = f"https://www.google.com/maps/dir/?api=1&origin={garage_loc[0]},{garage_loc[1]}&destination={DEONAR_DUMPING[0]},{DEONAR_DUMPING[1]}&waypoints=" + "|".join([f"{lat},{lon}" for lat, lon in zip(current_bins['lat'], current_bins['lon'])])
            st.image(qrcode.make(google_url).get_image(), width=180)
    except: st.error("Map Engine Initializing...")

# --- Impact & Financial Analysis (FULL RESTORED VERSION) ---
elif page == "Impact & Financial Analysis":
    st.title("💎 Comprehensive Business & Impact Model")
    st.markdown("### The 360° Value Proposition")
    
    st.sidebar.header("⚙️ Simulation Parameters")
    is_ev = st.sidebar.checkbox("⚡ Activate Electric Vehicle (EV) Fleet Mode")
    
    st.sidebar.subheader("1. CAPEX")
    num_trucks = st.sidebar.number_input("Fleet Size (Trucks)", value=5)
    hard_cost = st.sidebar.number_input("Hardware Cost/Bin (₹)", value=1500)
    total_bins_f = st.sidebar.number_input("Total Smart Bins", value=100)
    soft_cost = st.sidebar.number_input("Software Setup Cost (₹)", value=50000)
    
    st.sidebar.subheader("2. OPEX")
    wage = st.sidebar.slider("Staff Hourly Wage (₹)", 100, 500, 200)
    if is_ev:
        f_price, efficiency, c_factor = 10.0, 1.5, 0.82
    else:
        f_price, efficiency, c_factor = 104.0, 4.0, 2.68
    
    # Financial Engine Logic
    total_capex = (hard_cost * total_bins_f) + soft_cost
    
    # Scenario Comparison
    d_old, trips_old, h_old = 60.0, 30, 7.0
    d_new, trips_new, h_new = 40.0, 24, 5.0
    
    def calc_fin(d, t, h):
        total_d = d * t * num_trucks
        fuel = (total_d / efficiency) * f_price
        labor = (h * t * num_trucks) * wage
        return fuel + labor + (total_d * 5.0), (total_d / efficiency) * c_factor

    old_costs, old_co2 = calc_fin(d_old, trips_old, h_old)
    new_costs, new_co2 = calc_fin(d_new, trips_new, h_new)
    
    opex_savings = old_costs - new_costs
    co2_saved = (old_co2 - new_co2) / 1000
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Monthly Savings", f"₹{opex_savings:,.0f}")
    k2.metric("CO2 Prevented", f"{co2_saved:.1f} Tons")
    k3.metric("ROI Payback", f"{total_capex / (opex_savings + 5000):.1f} Mo")
    
    st.subheader("Value Breakdown")
    wf_df = pd.DataFrame({"Source": ["Fuel Savings", "Labor Efficiency", "CO2 Credits"], "Value": [opex_savings * 0.4, opex_savings * 0.5, 5000]})
    st.plotly_chart(px.bar(wf_df, x="Source", y="Value", color="Source"), width='stretch')
    
    st.subheader("Cumulative Cash Flow")
    cf = [-total_capex + ((opex_savings+5000) * i) for i in range(37)]
    st.plotly_chart(px.line(x=range(37), y=cf, labels={'x':'Month', 'y':'Balance'}).add_hline(y=0, line_dash="dash"), width='stretch')
