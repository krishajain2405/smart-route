import streamlit as st
import pandas as pd
import numpy as np
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

# --- 1. Page Configuration & Styling ---
st.set_page_config(page_title="Evergreen Smart Waste AI", page_icon="🚛", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Load and Cache Data ---
@st.cache_data
def load_data():
    # Automatically finds the CSV file in your directory
    target = 'smart_bin_historical_data.csv'
    if not os.path.exists(target):
        all_csvs = [f for f in os.listdir('.') if f.endswith('.csv')]
        target = all_csvs[0] if all_csvs else None
    
    if not target:
        st.error("CSV file not found. Please upload your data.")
        return None
        
    try:
        df = pd.read_csv(target, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower()
        # Mapping common names to standardized ones for the code logic
        df = df.rename(columns={
            'bin_location_lat': 'lat', 'bin_location_lon': 'lon',
            'bin_fill_percent': 'fill', 'timestamp': 'timestamp'
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        return df.dropna(subset=['timestamp'])
    except Exception as e:
        st.error(f"Load Error: {e}")
        return None

# --- 3. Global Constants ---
GARAGES = {
    "Truck 1 (Worli)": (19.0178, 72.8478),
    "Truck 2 (Bandra)": (19.0596, 72.8295),
    "Truck 3 (Andheri)": (19.1136, 72.8697),
    "Truck 4 (Kurla)": (19.0726, 72.8844),
    "Truck 5 (Borivali)": (19.2307, 72.8567)
}
DEONAR_DUMPING = (19.0550, 72.9250)

df = load_data()

if df is not None:
    # --- 4. Navigation Menu (Fixes NameError) ---
    st.sidebar.title("🌲 Mission Control")
    page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization"])

    # --- 5. Home Page ---
    if page == "Home":
        st.title("Smart Waste Management Analytics Dashboard")
        st.subheader("Project Overview")
        st.write("""
        - **Live Data:** Real-time fill-level tracking.
        - **Historical Analysis:** Waste generation pattern recognition.
        - **Predictive Modeling:** Random Forest forecasting.
        - **Multi-Trip Optimization:** AI-driven fleet routing.
        """)
        st.dataframe(df.head())

    # --- 6. EDA Page ---
    elif page == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis (EDA)")
        hourly_fill = df.groupby(['hour_of_day', 'area_type'])['fill'].mean().reset_index()
        fig1 = px.line(hourly_fill, x='hour_of_day', y='fill', color='area_type', title='Average Fill Level by Hour')
        st.plotly_chart(fig1, use_container_width=True)

    # --- 7. Predictive Model Page ---
    elif page == "Predictive Model":
        st.title("Predictive Model (Random Forest)")
        with st.spinner("Training Model..."):
            features = ['hour_of_day', 'day_of_week', 'ward', 'area_type', 'time_since_last_pickup']
            target = 'fill'
            model_df = pd.get_dummies(df[features + [target]].dropna(), columns=['day_of_week', 'ward', 'area_type'], drop_first=True)
            X = model_df.drop(target, axis=1)
            y = model_df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        
        c1, c2 = st.columns(2)
        c1.metric("MAE", f"{mean_absolute_error(y_test, preds):.2f}%")
        c2.metric("R² Score", f"{r2_score(y_test, preds):.2f}")

    # --- 8. Route Optimization Page (Fixes SyntaxError & Adds Multi-Trip) ---
    elif page == "Route Optimization":
        st.title("🚛 AI Multi-Fleet Route Optimizer")

        # Sidebar Controls
        selected_truck = st.sidebar.selectbox("Active Truck", list(GARAGES.keys()))
        threshold = st.sidebar.slider("Fill Threshold (%)", 0, 100, 75)
        
        # Simulation Slider (Fixes Threshold Visibility)
        times = sorted(df['timestamp'].unique())
        sim_time = st.sidebar.select_slider("Simulation Time", options=times, value=times[int(len(times)*0.85)])
        
        df_snap = df[df['timestamp'] == sim_time].copy()

        # Assignment Logic
        def assign_truck(row):
            loc = (row['lat'], row['lon'])
            dists = {name: ((loc[0]-c[0])**2 + (loc[1]-c[1])**2)**0.5 for name, c in GARAGES.items()}
            return min(dists, key=dists.get)

        df_snap['assigned_truck'] = df_snap.apply(assign_truck, axis=1)
        all_my_bins = df_snap[(df_snap['assigned_truck'] == selected_truck) & (df_snap['fill'] >= threshold)]
        all_my_bins = all_my_bins.sort_values('fill', ascending=False)

        # Greedy Multi-Trip Pipeline
        st.sidebar.markdown("---")
        bins_per_trip = 8 
        num_trips = (len(all_my_bins) // bins_per_trip) + (1 if len(all_my_bins) % bins_per_trip > 0 else 0)
        
        if num_trips > 0:
            trip_num = st.sidebar.selectbox(f"Select Trip (Total: {num_trips})", range(1, num_trips + 1), format_func=lambda x: f"Trip {x}")
            current_bins = all_my_bins.iloc[(trip_num-1)*bins_per_trip : trip_num*bins_per_trip]
        else:
            current_bins = pd.DataFrame()

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Vehicle", selected_truck)
        col2.metric("Pending Bins", len(all_my_bins))
        col3.metric("Current Trip Stops", len(current_bins))

        if not current_bins.empty:
            if st.button("🚀 Calculate Optimized Path"):
                with st.spinner("Running OR-Tools Solver..."):
                    garage_loc = GARAGES[selected_truck]
                    # Logic: Garage -> Bins -> Deonar
                    pts = [garage_loc] + list(zip(current_bins['lat'], current_bins['lon']))
                    
                    # Solver Engine
                    data = {'locations': pts, 'num_vehicles': 1, 'depot': 0}
                    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
                    routing = pywrapcp.RoutingModel(manager)
                    
                    def dist_callback(from_idx, to_idx):
                        p1, p2 = data['locations'][manager.IndexToNode(from_idx)], data['locations'][manager.IndexToNode(to_idx)]
                        return int(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 * 100000)
                    
                    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_callback))
                    solution = routing.SolveWithParameters(pywrapcp.DefaultRoutingSearchParameters())

                    if solution:
                        index, route_coords = routing.Start(0), []
                        while not routing.IsEnd(index):
                            route_coords.append(data['locations'][manager.IndexToNode(index)])
                            index = solution.Value(routing.NextVar(index))
                        route_coords.append(data['locations'][manager.IndexToNode(index)])
                        route_coords.append(DEONAR_DUMPING)

                        # Map & QR
                        m = folium.Map(location=[19.0760, 72.8777], zoom_start=12, tiles="CartoDB positron")
                        folium.PolyLine(route_coords, color="blue", weight=5).add_to(m)
                        folium.Marker(garage_loc, icon=folium.Icon(color='red', icon='truck')).add_to(m)
                        for _, r in current_bins.iterrows():
                            folium.Marker([r['lat'], r['lon']], icon=folium.Icon(color='blue', icon='trash')).add_to(m)
                        st_folium(m, width=1100, height=500, key="optimized_map")
                        
                        st.subheader("📲 Driver Navigation QR")
                        url = f"https://www.google.com/maps/dir/?api=1&origin={garage_loc[0]},{garage_loc[1]}&destination={DEONAR_DUMPING[0]},{DEONAR_DUMPING[1]}"
                        st.image(qrcode.make(url).get_image(), width=200)
        else:
            st.info("No bins meet the threshold. Adjust simulation time or threshold slider.")
