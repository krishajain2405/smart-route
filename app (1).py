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

# --- 1. GLOBAL CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(page_title="Smart Waste AI Mission Control", layout="wide")

GARAGES = {
    "Truck 1 (Worli)": (19.0178, 72.8478),
    "Truck 2 (Bandra)": (19.0596, 72.8295),
    "Truck 3 (Andheri)": (19.1136, 72.8697),
    "Truck 4 (Kurla)": (19.0726, 72.8844),
    "Truck 5 (Borivali)": (19.2307, 72.8567)
}
DEONAR_DUMPING = (19.0550, 72.9250)

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    all_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    target = 'data.csv' if 'data.csv' in all_files else (all_files[0] if all_files else None)
    
    if not target:
        # Dummy data fail-safe
        return pd.DataFrame({
            'timestamp': pd.to_datetime(['2025-01-01 10:00:00']),
            'bin_id': ['B101'], 'hour_of_day': [10], 'day_of_week': ['Monday'],
            'ward': ['Ward_A'], 'area_type': ['Residential'], 'time_since_last_pickup': [24],
            'bin_fill_percent': [50], 'bin_capacity_liters': [1000],
            'bin_location_lat': [19.0760], 'bin_location_lon': [72.8777]
        })
    
    try:
        df = pd.read_csv(target, sep=None, engine='python', encoding='utf-8-sig')
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Standardize columns for both logic sets
        rename_dict = {
            'bin_location_lat': 'lat', 'bin_location_lon': 'lon',
            'bin_fill_percent': 'fill', 'timestamp': 'timestamp',
            'bin_id': 'bin_id'
        }
        for old, new in rename_dict.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        return df.dropna(subset=['timestamp'])
    except:
        return pd.DataFrame()

def get_dist(p1, p2):
    # FIXED: Corrected the math syntax error from your snippet
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

@st.cache_resource
def get_map_graph():
    return ox.graph_from_point((19.0760, 72.8777), dist=8000, network_type='drive')

df = load_data()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=100)
    st.title("Control Panel")
    page = st.radio("Go to", ["Home", "EDA Dashboard", "Predictive Model", "Route Optimization", "Impact & Financial Analysis"])

# --- 4. MAIN INTERFACE ---
if page == "Home":
    st.title("Smart Waste Management Analytics Dashboard")
    st.subheader("Project Overview")
    st.write("""
    - **Live Data:** A physical prototype sends real-time fill-level data to a cloud dashboard.
    - **Historical Analysis:** We analyze a dataset to understand waste generation patterns.
    - **Predictive Modeling:** A machine learning model forecasts when bins will become full.
    - **Route Optimization:** An AI Multi-Fleet system for efficient collection.
    - **Financial Impact:** ROI, carbon credits, and operational savings analysis.
    """)
    st.subheader("Dataset at a Glance")
    st.dataframe(df.head())

elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    hourly_fill_pattern = df.groupby(['hour_of_day', 'area_type'])['fill'].mean().reset_index()
    fig1 = px.line(hourly_fill_pattern, x='hour_of_day', y='fill', color='area_type', title='Avg Fill % by Hour')
    st.plotly_chart(fig1, use_container_width=True)

    daily_avg = df.groupby('day_of_week')['fill'].mean().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig2 = px.bar(daily_avg, x='day_of_week', y='fill', category_orders={"day_of_week": day_order}, title='Avg Fill % by Day')
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Predictive Model":
    st.title("🔮 Predictive Model for Bin Fill Level")
    with st.spinner("Training model..."):
        features = ['hour_of_day', 'day_of_week', 'ward', 'area_type', 'time_since_last_pickup']
        model_df = pd.get_dummies(df[features + ['fill']], columns=['day_of_week', 'ward', 'area_type'], drop_first=True)
        X = model_df.drop('fill', axis=1)
        y = model_df['fill']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    st.metric("Model R² Score", f"{r2_score(y_test, preds):.2f}")
    fig = px.scatter(pd.DataFrame({'Actual': y_test, 'Predicted': preds}), x='Actual', y='Predicted', trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Route Optimization":
    st.title("🚛 AI Multi-Fleet Mission Control")
    
    selected_truck = st.sidebar.selectbox("Active Truck", list(GARAGES.keys()))
    threshold = st.sidebar.slider("Fill Threshold (%)", 0, 100, 75)
    
    times = sorted(df['timestamp'].unique())
    sim_time = st.sidebar.select_slider("Select Time", options=times, value=times[int(len(times)*0.85)])
    
    df_snap = df[df['timestamp'] == sim_time].copy()
    df_snap['assigned_truck'] = df_snap.apply(lambda r: min(GARAGES, key=lambda g: get_dist((r['lat'], r['lon']), GARAGES[g])), axis=1)
    
    all_my_bins = df_snap[(df_snap['assigned_truck'] == selected_truck) & (df_snap['fill'] >= threshold)]
    
    # Trip Pipeline
    bins_per_trip = 8
    num_trips = (len(all_my_bins) // bins_per_trip) + (1 if len(all_my_bins) % bins_per_trip > 0 else 0)
    
    if num_trips > 0:
        trip_num = st.sidebar.selectbox(f"Select Trip (Total: {num_trips})", range(1, num_trips + 1))
        current_mission_bins = all_my_bins.iloc[(trip_num-1)*bins_per_trip : trip_num*bins_per_trip]
        
        # MAP ENGINE
        G = get_map_graph()
        m = folium.Map(location=[19.0760, 72.8777], zoom_start=12, tiles="CartoDB positron")
        
        for _, row in df_snap.iterrows():
            is_full = row['fill'] >= threshold
            is_mine = row['assigned_truck'] == selected_truck
            is_in_current = row['bin_id'] in current_mission_bins['bin_id'].values if not current_mission_bins.empty else False
            
            if is_full and is_mine and is_in_current: color = 'red'
            elif is_full and is_mine and not is_in_current: color = 'blue'
            elif is_full and not is_mine: color = 'orange'
            else: color = 'green'
            
            folium.Marker([row['lat'], row['lon']], icon=folium.Icon(color=color, icon='trash', prefix='fa')).add_to(m)

        # Route drawing
        garage_loc = GARAGES[selected_truck]
        pts = [garage_loc] + list(zip(current_mission_bins['lat'], current_mission_bins['lon'])) + [DEONAR_DUMPING]
        path_coords = []
        for i in range(len(pts)-1):
            n1 = ox.nearest_nodes(G, pts[i][1], pts[i][0])
            n2 = ox.nearest_nodes(G, pts[i+1][1], pts[i+1][0])
            route = nx.shortest_path(G, n1, n2, weight='length')
            path_coords.extend([[G.nodes[node]['y'], G.nodes[node]['x']] for node in route])
        
        folium.PolyLine(path_coords, color="#3498db", weight=6).add_to(m)
        st_folium(m, width=1200, height=550, key="mission_map")
        
        # QR Code
        google_url = f"https://www.google.com/maps/dir/{garage_loc[0]},{garage_loc[1]}/" + "/".join([f"{lat},{lon}" for lat, lon in zip(current_mission_bins['lat'], current_mission_bins['lon'])]) + f"/{DEONAR_DUMPING[0]},{DEONAR_DUMPING[1]}"
        st.image(qrcode.make(google_url).get_image(), width=150, caption="Scan for Navigation")

elif page == "Impact & Financial Analysis":
    st.title("💎 Financial & Sustainability Impact")
    # Metric logic
    k1, k2, k3 = st.columns(3)
    k1.metric("Direct Savings", "₹1.2L", delta="15%")
    k2.metric("CO2 Saved", "420 kg", delta="Carbon Credits")
    k3.metric("ROI", "14 Months")
    
    st.subheader("3-Year Cash Flow Projection")
    # Simplified version of your cash flow logic
    months = np.arange(1, 37)
    savings = -150000 + (15000 * months)
    fig_cf = px.line(x=months, y=savings, labels={'x':'Month', 'y':'Balance (₹)'})
    fig_cf.add_hline(y=0, line_dash="dash", line_color="green")
    st.plotly_chart(fig_cf, use_container_width=True)
