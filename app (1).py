import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Smart Bin AI", layout="wide")

# --- STEP 1: UPLOAD DATA ---
st.sidebar.title("📁 Mission Control")
uploaded_file = st.sidebar.file_uploader("Upload data.csv to start", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Clean column names immediately
    df.columns = df.columns.str.strip().str.lower()
    
    # Map your specific columns to standard names
    mapping = {
        'bin_fill_percent': 'fill', 'bin_location_lat': 'lat', 
        'bin_location_lon': 'lon', 'hour_of_day': 'hour',
        'time_since_last_pickup': 'tslp'
    }
    for old, new in mapping.items():
        if old in df.columns: df[new] = df[old]

    # --- STEP 2: NAVIGATION ---
    page = st.sidebar.radio("Navigate", ["Home", "EDA", "Predictive AI", "Map", "Financials"])

    if page == "Home":
        st.title("🚛 Smart Waste Management Dashboard")
        st.success(f"Successfully loaded {len(df)} records.")
        st.dataframe(df)

    elif page == "EDA":
        st.title("📊 Waste Generation Patterns")
        if 'hour' in df.columns and 'fill' in df.columns:
            fig = px.line(df.groupby('hour')['fill'].mean().reset_index(), x='hour', y='fill')
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Predictive AI":
        st.title("🤖 AI Forecasting")
        try:
            X = df[['hour', 'tslp']] if 'tslp' in df.columns else df[['hour']]
            y = df['fill']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestRegressor().fit(X_train, y_train)
            st.metric("Model Confidence", f"{model.score(X_test, y_test)*100:.1f}%")
        except:
            st.error("Missing prediction columns in CSV.")

    elif page == "Map":
        st.title("📍 Dispatch Map")
        m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=12)
        for _, r in df.head(100).iterrows():
            folium.Marker([r['lat'], r['lon']], icon=folium.Icon(color='red')).add_to(m)
        st_folium(m, width=1000, height=500)

    elif page == "Financials":
        st.title("💎 ROI & Sustainability")
        trucks = st.sidebar.number_input("Trucks", value=5)
        st.metric("Monthly Savings", f"₹{trucks * 15000:,.0f}")
        st.write("Analysis based on your custom fleet parameters.")

else:
    st.title("🚛 Smart Bin Mission Control")
    st.info("Awaiting Data: Please upload your 'data.csv' in the sidebar to begin the presentation.")
