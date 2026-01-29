# app.py - Complete Integrated Smart Waste Management System
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import time
import json
import base64
import requests
from io import BytesIO
from PIL import Image
import random

# Machine Learning & Optimization
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Firebase Integration
import firebase_admin
from firebase_admin import credentials, firestore, db
from firebase_admin.exceptions import FirebaseError

# AI/ML for Image Classification
try:
    from transformers import pipeline
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False

# ==========================================
# 🎨 PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Smart Waste Monitoring System",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching your screenshot design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header Gradient */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00C853 0%, #1DE9B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    /* Sub Header */
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        margin: 10px 0 0 0;
        font-weight: 500;
    }
    
    /* Status Indicators */
    .status-normal {
        color: #00C853;
        font-weight: 600;
        background: rgba(0, 200, 83, 0.1);
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    
    .status-warning {
        color: #FF9800;
        font-weight: 600;
        background: rgba(255, 152, 0, 0.1);
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    
    .status-critical {
        color: #FF5252;
        font-weight: 600;
        background: rgba(255, 82, 82, 0.1);
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    
    /* Alert Cards */
    .alert-card {
        border-left: 5px solid #FF5252;
        background: linear-gradient(90deg, #FFF5F5 0%, #FFEBEE 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid #f0f0f0;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .gradient-btn {
        background: linear-gradient(90deg, #00C853 0%, #1DE9B6 100%);
        color: white;
        border: none;
        padding: 14px 32px;
        border-radius: 25px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .gradient-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,200,83,0.25);
    }
    
    /* Driver Cards */
    .driver-card {
        background: linear-gradient(135deg, #4A6FA5 0%, #6B8CBC 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Leaderboard */
    .leaderboard-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        margin: 10px 0;
        border: 2px solid #f0f0f0;
    }
    
    .gold {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: white;
        border: none !important;
    }
    
    .silver {
        background: linear-gradient(135deg, #C0C0C0 0%, #A0A0A0 100%);
        color: white;
        border: none !important;
    }
    
    .bronze {
        background: linear-gradient(135deg, #CD7F32 0%, #A0522D 100%);
        color: white;
        border: none !important;
    }
    
    /* Progress Bars */
    .progress-container {
        background: #e0e0e0;
        border-radius: 10px;
        height: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Map Container */
    .map-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #00C853;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🔧 CONFIGURATION & INITIALIZATION
# ==========================================
# ==========================================

# Check if Firebase is available


FIREBASE_URL =   "https://smart-bin-7efab-default-rtdb.firebaseio.com"
HF_API_KEY ="AIzaSyAfDSZS8t7tNm2C9y8YCR-N_KTMQ5kKdUw"
AI_MODEL_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14"
# AI Verification (Infinity Logic)
def verify_image(image_bytes):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = requests.post(AI_MODEL_URL, headers=headers, data=image_bytes)
        data = response.json()
        return data['labels'][0], data['scores'][0] 
    except:
        return "Error", 0.0

# Fetch Live Data (Firebase)
from datetime import datetime

def fetch_live_data():
    try:
        r = requests.get(f"{FIREBASE_URL}/bins.json", timeout=10)
        data = r.json()

        bins = []

        if not data:
            return bins

        for bin_id, info in data.items():
            fill = int(info.get("fill_level", 0))

            if fill >= 85:
                status = "Critical"
            elif fill >= 60:
                status = "Warning"
            else:
                status = "Normal"

            bins.append({
                "bin_id": bin_id,
                "location": (info.get("lat"), info.get("lon")),
                "fill_level": fill,
                "weight_kg": info.get("weight_kg", 0),
                "status": status,
                "last_updated": datetime.now()  # ✅ ADD THIS
            })

        return bins

    except Exception as e:
        st.error("❌ Firebase fetch failed")
        return []



# Parbhani Coordinates (Vasant Naik College)
Parbhani_COORDS = {
    'lat': 19.2335,
    'lon': 76.7845
}
# ==========================================
# 🚀 DATA MANAGEMENT FUNCTIONS
# ==========================================


@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("smart_bin_historical_data.csv")

        # SAFELY parse timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        return df

    except FileNotFoundError:
        st.error("❌ smart_bin_historical_data.csv not found")
        return pd.DataFrame()


    except FileNotFoundError:
        # Create sample data if file doesn't exist
        st.warning("Using sample data. For real data, add 'smart_bin_historical_data.csv' to your directory.")
        
        # Generate comprehensive sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
        bins = [f'Bin_{i:03d}' for i in range(1, 26)]
        
        data = []
        for date in dates:
            for bin_id in bins:
                hour = date.hour
                day_of_week = date.strftime('%A')
                
                # Generate realistic fill levels based on time patterns
                base_fill = 30
                hour_effect = np.sin(hour * np.pi / 12) * 20
                day_effect = 10 if day_of_week in ['Saturday', 'Sunday'] else 0
                noise = np.random.normal(0, 10)
                
                fill_level = max(0, min(100, base_fill + hour_effect + day_effect + noise))
                
                data.append({
                    'timestamp': date,
                    'bin_id': bin_id,
                    'hour_of_day': hour,
                    'day_of_week': day_of_week,
                    'ward': f'Ward_{(ord(bin_id[-3]) % 5) + 1}',
                    'area_type': np.random.choice(['Residential', 'Commercial', 'Public', 'Industrial'], p=[0.6, 0.2, 0.15, 0.05]),
                    'time_since_last_pickup': np.random.randint(1, 72),
                    'bin_fill_percent': fill_level,
                    'bin_capacity_liters': np.random.choice([120, 240, 360, 1000]),
                    'bin_location_lat': Parbhani_COORDS['lat'] + np.random.uniform(-0.02, 0.02),
                    'bin_location_lon': Parbhani_COORDS['lon'] + np.random.uniform(-0.02, 0.02),
                    'temperature': np.random.uniform(20, 35),
                    'humidity': np.random.uniform(40, 80),
                    'waste_type': np.random.choice(['Organic', 'Recyclable', 'Mixed'], p=[0.5, 0.3, 0.2])
                })
        
        df = pd.DataFrame(data)
        return df

def generate_realtime_bin_data():
    """Generate real-time bin data (simulating ESP32 updates)"""
    bins = []
    
    # Create bins around Parbhani
    locations = [
        (Parbhani_COORDS['lat'] + 0.001, Parbhani_COORDS['lon'] + 0.001),
        (Parbhani_COORDS['lat'] + 0.002, Parbhani_COORDS['lon'] - 0.001),
        (Parbhani_COORDS['lat'] - 0.001, Parbhani_COORDS['lon'] + 0.002),
        (Parbhani_COORDS['lat'] - 0.002, Parbhani_COORDS['lon'] - 0.002),
        (Parbhani_COORDS['lat'] + 0.003, Parbhani_COORDS['lon'] + 0.003),
        (Parbhani_COORDS['lat'] - 0.003, Parbhani_COORDS['lon'] + 0.001),
    ]
    
    for i, (lat, lon) in enumerate(locations):
        fill_level = random.randint(10, 95)
        if fill_level < 60:
            status = 'Normal'
        elif fill_level < 85:
            status = 'Warning'
        else:
            status = 'Critical'
        
        bins.append({
            'bin_id': f'Bin_{101 + i}',
            'location': (lat, lon),
            'fill_level': fill_level,
            'status': status,
            'last_updated': datetime.now() - timedelta(minutes=random.randint(1, 30)),
            'weight_kg': random.randint(5, 50),
            'address': f'Vasant Naik College Area, Zone {i+1}',
            'capacity_liters': random.choice([120, 240, 360])
        })
    
    return bins

def generate_driver_locations():
    """Generate driver locations around Parbhani"""
    drivers = []
    
    base_lat, base_lon = Parbhani_COORDS['lat'], Parbhani_COORDS['lon']
    
    driver_locations = [
        (base_lat + 0.005, base_lon + 0.005),
        (base_lat - 0.005, base_lon + 0.003),
        (base_lat + 0.003, base_lon - 0.005),
        (base_lat - 0.003, base_lon - 0.003),
        (base_lat + 0.006, base_lon),
    ]
    
    names = ['Rajesh Kumar', 'Amit Sharma', 'Vikram Singh', 'Sanjay Patel', 'Deepak Verma']
    phones = ['+91 98765 43210', '+91 98765 43211', '+91 98765 43212', '+91 98765 43213', '+91 98765 43214']
    
    for i, (lat, lon) in enumerate(driver_locations):
        drivers.append({
            'driver_id': f'DRV-{1000 + i}',
            'name': names[i],
            'location': (lat, lon),
            'status': random.choice(['Available', 'On Route', 'Available', 'Available']),
            'phone': phones[i],
            'vehicle_type': 'Electric Truck' if i % 2 == 0 else 'Diesel Truck',
            'rating': round(random.uniform(4.0, 5.0), 1)
        })
    
    return drivers

# ==========================================
# 🧠 AI & MACHINE LEARNING FUNCTIONS
# ==========================================

@st.cache_resource
def load_ai_model():
    """Load Hugging Face garbage classification model"""
    if AI_AVAILABLE:
        try:
            return pipeline("image-classification", model="yangy50/garbage-classification")
        except:
            return None
    return None

def classify_garbage_image(image_file):
    """Classify garbage image using AI model"""
    model = load_ai_model()
    
    if model:
        try:
            image = Image.open(image_file)
            results = model(image)
            return {
                'predictions': results,
                'primary_category': results[0]['label'],
                'confidence': results[0]['score']
            }
        except Exception as e:
            st.warning(f"AI model error: {str(e)}")
    
    # Fallback to simulated classification
    categories = ['Organic', 'Recyclable', 'Plastic', 'Paper', 'Glass', 'Metal', 'Hazardous']
    probabilities = np.random.dirichlet(np.ones(len(categories)))
    
    return {
        'predictions': [
            {'label': cat, 'score': float(prob)}
            for cat, prob in zip(categories, probabilities)
        ],
        'primary_category': categories[np.argmax(probabilities)],
        'confidence': float(max(probabilities))
    }

@st.cache_resource
def train_predictive_model(historical_data):
    """Train Random Forest model for fill level prediction"""
    try:
        # Prepare features
        df = historical_data.copy()
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_of_week_num'] = pd.Categorical(df['day_of_week'], 
                                              categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).codes
        
        # Encode categorical variables
        df = pd.get_dummies(df, columns=['area_type', 'ward'], drop_first=True)
        
        # Features and target
        feature_cols = ['hour_sin', 'hour_cos', 'day_of_week_num', 'time_since_last_pickup', 
                       'temperature', 'humidity'] + [col for col in df.columns if 'area_type_' in col or 'ward_' in col]
        
        X = df[feature_cols].fillna(0)
        y = df['bin_fill_percent']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, mae, r2, X_train.columns.tolist()
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return None, None, None, None

# ==========================================
# 🗺️ ROUTE OPTIMIZATION FUNCTIONS
# ==========================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate approximate distance between two coordinates (in km)"""
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  # Rough conversion to km

def optimize_route_ortools(bins_to_collect, depot_location, num_vehicles=2):
    """Optimize collection routes using OR-Tools"""
    try:
        # Add depot as first location
        locations = [depot_location] + [bin['location'] for bin in bins_to_collect]
        
        # Create distance matrix
        distance_matrix = []
        for i in range(len(locations)):
            row = []
            for j in range(len(locations)):
                if i == j:
                    row.append(0)
                else:
                    dist = calculate_distance(locations[i][0], locations[i][1], 
                                            locations[j][0], locations[j][1])
                    row.append(int(dist * 1000))  # Convert to meters for precision
            distance_matrix.append(row)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add capacity constraints (if needed)
        demands = [0] + [1] * len(bins_to_collect)  # Depot has 0 demand
        vehicle_capacities = [len(bins_to_collect)] * num_vehicles
        
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            vehicle_capacities,
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(2)
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            routes = []
            total_distance = 0
            
            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                route = []
                route_distance = 0
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index != 0:  # Skip depot
                        route.append(bins_to_collect[node_index - 1])
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
                
                if route:
                    routes.append({
                        'vehicle_id': vehicle_id,
                        'bins': route,
                        'distance_km': route_distance / 1000,
                        'estimated_time_min': len(route) * 10 + route_distance / 1000 * 2
                    })
                    total_distance += route_distance / 1000
            
            return routes, total_distance
        
    except Exception as e:
        st.error(f"Route optimization error: {str(e)}")
    
    return [], 0

def find_nearest_driver(bin_location, drivers):
    """Find nearest available driver to bin"""
    available_drivers = [d for d in drivers if d['status'] == 'Available']
    
    if not available_drivers:
        return None
    
    # Find driver with minimum distance
    min_distance = float('inf')
    nearest_driver = None
    
    for driver in available_drivers:
        dist = calculate_distance(bin_location[0], bin_location[1], 
                                 driver['location'][0], driver['location'][1])
        if dist < min_distance:
            min_distance = dist
            nearest_driver = driver
    
    nearest_driver['distance_to_bin'] = min_distance
    return nearest_driver

# ==========================================
# 💰 FINANCIAL MODEL FUNCTIONS
# ==========================================

def calculate_financial_roi(params):
    """Calculate financial ROI based on parameters"""
    # CAPEX
    capex = (params['hardware_cost_per_bin'] * params['total_bins']) + params['software_dev_cost']
    
    # OPEX for traditional system
    old_opex = {
        'fuel': params['dist_old'] * params['trips_old'] * params['num_trucks'] * params['fuel_price'] / params['truck_efficiency'],
        'labor': params['hours_old'] * params['trips_old'] * params['num_trucks'] * params['driver_wage'],
        'maintenance': params['dist_old'] * params['trips_old'] * params['num_trucks'] * params['maintenance_per_km']
    }
    old_total_opex = sum(old_opex.values())
    
    # OPEX for smart system
    new_opex = {
        'fuel': params['dist_new'] * params['trips_new'] * params['num_trucks'] * params['fuel_price'] / params['truck_efficiency'],
        'labor': params['hours_new'] * params['trips_new'] * params['num_trucks'] * params['driver_wage'],
        'maintenance': params['dist_new'] * params['trips_new'] * params['num_trucks'] * params['maintenance_per_km'],
        'cloud': params['cloud_cost_per_bin'] * params['total_bins']
    }
    new_total_opex = sum(new_opex.values())
    
    # Savings
    opex_savings = old_total_opex - new_total_opex
    
    # Revenue from recycling
    base_revenue = params['daily_waste_collected_kg'] * 30 * params['recyclable_value_per_kg'] * 0.1
    improved_revenue = params['daily_waste_collected_kg'] * 30 * params['recyclable_value_per_kg'] * (0.1 + params['recycling_rate_increase']/100)
    revenue_gain = improved_revenue - base_revenue
    
    # Penalty avoidance
    penalty_savings = params['overflows_prevented_month'] * params['penalty_per_overflow']
    
    # Carbon credits
    co2_saved_tons = ((old_opex['fuel'] / params['fuel_price'] * params['co2_factor']) - 
                     (new_opex['fuel'] / params['fuel_price'] * params['co2_factor'])) / 1000
    carbon_credit_revenue = co2_saved_tons * params['carbon_credit_price']
    
    # Total monthly benefit
    total_monthly_benefit = opex_savings + revenue_gain + penalty_savings + carbon_credit_revenue
    
    # ROI
    months_breakeven = capex / total_monthly_benefit if total_monthly_benefit > 0 else 0
    
    return {
        'capex': capex,
        'old_opex': old_total_opex,
        'new_opex': new_total_opex,
        'opex_savings': opex_savings,
        'revenue_gain': revenue_gain,
        'penalty_savings': penalty_savings,
        'carbon_credit_revenue': carbon_credit_revenue,
        'total_monthly_benefit': total_monthly_benefit,
        'months_breakeven': months_breakeven,
        'co2_saved_tons': co2_saved_tons,
        'old_opex_breakdown': old_opex,
        'new_opex_breakdown': new_opex
    }

# ==========================================
# 🎮 GAMIFICATION FUNCTIONS
# ==========================================

def generate_leaderboard():
    """Generate citizen leaderboard with green points"""
    citizens = [
        {'name': 'Eco Warrior', 'points': 1250, 'level': 'Platinum', 'reports': 45},
        {'name': 'Green Guardian', 'points': 980, 'level': 'Gold', 'reports': 38},
        {'name': 'Clean Champion', 'points': 750, 'level': 'Silver', 'reports': 32},
        {'name': 'Waste Warrior', 'points': 620, 'level': 'Bronze', 'reports': 28},
        {'name': 'Recycle Master', 'points': 540, 'level': 'Bronze', 'reports': 25},
        {'name': 'Sustainability Hero', 'points': 480, 'level': 'Bronze', 'reports': 22},
        {'name': 'Earth Protector', 'points': 420, 'level': 'Bronze', 'reports': 20},
        {'name': 'Eco Pioneer', 'points': 380, 'level': 'Bronze', 'reports': 18},
        {'name': 'Green Innovator', 'points': 320, 'level': 'Bronze', 'reports': 16},
        {'name': 'Clean Visionary', 'points': 280, 'level': 'Bronze', 'reports': 14},
    ]
    return citizens

# ==========================================
# 📱 WHATSAPP INTEGRATION
# ==========================================

def generate_whatsapp_url(phone, message):
    """Generate WhatsApp URL with pre-filled message"""
    encoded_message = requests.utils.quote(message)
    return f"https://wa.me/{phone}?text={encoded_message}"

# ==========================================
# 🏠 LANDING PAGE
# ==========================================

def show_landing_page():
    """Show professional landing page matching screenshots"""
    
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">IoT-based Smart Waste Monitoring System</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Transform Waste Collection with Real-Time Intelligence</p>', unsafe_allow_html=True)
        st.markdown("""
        Eliminate overflowing bins and optimize collection routes with our ESP32-powered 
        monitoring system. Get predictive insights and automated alerts that reduce 
        costs and improve sustainability.
        """)
        
        # Call to Action Buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🚀 See Live Dashboard", use_container_width=True, key="btn_dashboard"):
                st.session_state.page = "📊 Real-Time Monitoring"
                st.rerun()
        with btn_col2:
            if st.button("✨ Explore Features", use_container_width=True, key="btn_features"):
                st.session_state.page = "🔍 System Features"
                st.rerun()
    
    with col2:
        # Animated placeholder for waste management
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white;">
            <h2 style="color: white;">Live IoT Dashboard</h2>
            <p style="font-size: 3rem;">🗑️</p>
            <p>Real-time monitoring from<br>ESP32 sensors in Parbhani</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Key Metrics Section
    st.subheader("📊 Real-Time Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 class="metric-value">24</h3>
            <p class="metric-label">Active Bins Monitored</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 class="metric-value">3</h3>
            <p class="metric-label">Bins Need Collection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 class="metric-value">68%</h3>
            <p class="metric-label">Average Fill Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 class="metric-value">12</h3>
            <p class="metric-label">Scheduled Pickups Today</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # The Challenge Section
    st.markdown('<h2 class="section-header">THE CHALLENGE</h2>', unsafe_allow_html=True)
    st.markdown("""
    ### Traditional Waste Management is Broken
    
    Inefficient collection routes, overflowing bins, and reactive maintenance waste time, 
    money, and resources while harming the environment.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🚛 Inefficient Routes</h4>
            <p>Collection trucks visit half-empty bins while others overflow, wasting fuel and time.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>⚠️ Reactive Maintenance</h4>
            <p>Staff only respond after bins overflow, leading to complaints and environmental issues.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 No Data Insights</h4>
            <p>Without real-time data, it's impossible to optimize operations or predict future needs.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Core Technology Section
    st.markdown('<h2 class="section-header">CORE TECHNOLOGY</h2>', unsafe_allow_html=True)
    st.markdown("### Intelligent System Components")
    st.markdown("Our IoT solution combines cutting-edge hardware and cloud analytics to deliver actionable insights for smarter waste management.")
    
    tech_features = [
        {"icon": "🔌", "title": "ESP32 Microcontroller", "desc": "Powerful, energy-efficient processor for data collection"},
        {"icon": "📡", "title": "Ultrasonic Sensors", "desc": "Precisely measure bin fill levels in real-time"},
        {"icon": "⚖️", "title": "Load Cell Sensors", "desc": "Track waste weight for composition insights"},
        {"icon": "☁️", "title": "Arduino IoT Cloud", "desc": "Secure cloud infrastructure for data processing"},
        {"icon": "🤖", "title": "Predictive Models", "desc": "ML algorithms forecast fill times accurately"},
        {"icon": "🗺️", "title": "Route Optimization", "desc": "OR-Tools algorithm saves 30% on fuel costs"},
    ]
    
    cols = st.columns(3)
    for i, feature in enumerate(tech_features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <h2 style="font-size: 2.5rem; margin-bottom: 10px;">{feature['icon']}</h2>
                <h4 style="margin: 10px 0;">{feature['title']}</h4>
                <p style="color: #666; line-height: 1.5;">{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # CTA Section
    st.markdown('<h2 class="section-header">Ready to Transform Your Waste Management?</h2>', unsafe_allow_html=True)
    st.markdown("""
    Join the data-driven revolution in waste collection. Get real-time insights, 
    predictive analytics, and optimized routes that reduce costs and improve sustainability.
    """)
    
    cta_col1, cta_col2, cta_col3 = st.columns(3)
    
    with cta_col1:
        if st.button("📧 Get In Touch", use_container_width=True):
            st.info("📧 Contact: info@smartwaste.io")
    with cta_col2:
        if st.button("📞 Request Demo", use_container_width=True):
            st.info("📞 Call: +1 (555) 123-4567")
    with cta_col3:
        if st.button("📚 Learn More", use_container_width=True):
            st.session_state.page = "🔍 System Features"
            st.rerun()

# ==========================================
# 📊 REAL-TIME MONITORING DASHBOARD
# ==========================================

from datetime import datetime

def show_realtime_monitoring():
    st.title("🌍 Real-Time Monitoring Dashboard")

    # -------------------------------
    # SAFE DEFAULTS
    # -------------------------------
    Parbhani_COORDS = {"lat": 19.2686, "lon": 76.7708}

    bins = fetch_live_data() or []

    drivers = [
        {
            "driver_id": "driver01",
            "name": "Ramesh",
            "location": (19.2700, 76.7720),
            "status": "Available",
            "phone": "919999999999"
        },
        {
            "driver_id": "driver02",
            "name": "Suresh",
            "location": (19.2650, 76.7680),
            "status": "Busy",
            "phone": "919888888888"
        }
    ]

    # -------------------------------
    # TOP METRICS
    # -------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Bins", len(bins))

    with col2:
        st.metric("Need Collection", len([b for b in bins if b["status"] == "Critical"]))

    with col3:
        avg_fill = np.mean([b["fill_level"] for b in bins]) if bins else 0
        st.metric("Avg Fill Level", f"{avg_fill:.1f}%")

    with col4:
        st.metric("Scheduled Pickups", "12")

    st.divider()

    # -------------------------------
    # MAP
    # -------------------------------
    st.subheader("📍 Live Bin Locations – Parbhani")

    m = folium.Map(
        location=[Parbhani_COORDS["lat"], Parbhani_COORDS["lon"]],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    for b in bins:
        status = b["status"]
        color = "green" if status == "Normal" else "orange" if status == "Warning" else "red"

        folium.Marker(
            location=b["location"],
            popup=f"""
            <b>{b['bin_id']}</b><br>
            Status: {status}<br>
            Fill Level: {b['fill_level']}%<br>
            Weight: {b['weight_kg']} kg
            """,
            icon=folium.Icon(color=color, icon="trash")
        ).add_to(m)

    st_folium(m, width=800, height=500)

    st.divider()

    # -------------------------------
    # ACTIVE ALERTS
    # -------------------------------
    st.subheader("🚨 Active Alerts")

    critical_bins = [b for b in bins if b["status"] == "Critical"]

    if critical_bins:
        for b in critical_bins:
            minutes_ago = (datetime.now() - b["last_updated"]).seconds // 60
            st.warning(
                f"⚠️ Bin {b['bin_id']} is {b['fill_level']}% full "
                f"(updated {minutes_ago} min ago)"
            )
    else:
        st.success("✅ No critical alerts")

    st.divider()

    # -------------------------------
    # ROUTE OPTIMIZATION
    # -------------------------------
    st.subheader("🚛 Optimized Collection Routes")

    bins_to_collect = [b for b in bins if b["fill_level"] > 70]

    if bins_to_collect:
        depot = (Parbhani_COORDS["lat"], Parbhani_COORDS["lon"])
        routes, total_distance = optimize_route_ortools(bins_to_collect, depot)

        for r in routes:
            st.markdown(
                f"""
                **Vehicle {r['vehicle_id']}**  
                Distance: {r['distance_km']:.2f} km  
                ETA: {r['estimated_time_min']} mins  
                """
            )

        st.metric("Total Optimized Distance", f"{total_distance:.2f} km")
    else:
        st.info("No bins exceed collection threshold")

    st.divider()

    # -------------------------------
    # EMERGENCY DISPATCH
    # -------------------------------
    st.subheader("🚨 Emergency Dispatch System")

    emergency_bins = [b for b in bins if b["fill_level"] > 85]

    if emergency_bins:
        for eb in emergency_bins:
            st.error(f"EMERGENCY: Bin {eb['bin_id']} at {eb['fill_level']}%")

            nearest_driver = find_nearest_driver(eb["location"], drivers)

            if nearest_driver:
                st.success(
                    f"Assigned Driver: {nearest_driver['name']} "
                    f"({nearest_driver['driver_id']})"
                )

                if st.button(f"📱 Alert {nearest_driver['name']}", key=eb["bin_id"]):
                    st.info("WhatsApp alert simulated")
            else:
                st.error("No available drivers")
    else:
        st.success("✅ No emergency bins")


# ==========================================
# 🚚 DRIVER PORTAL
# ==========================================

def show_driver_portal():
    """Driver login and dispatch portal"""
    st.title("🚚 Driver Portal")
    
    # Initialize session state for driver authentication
    if 'driver_logged_in' not in st.session_state:
        st.session_state.driver_logged_in = False
    if 'driver_info' not in st.session_state:
        st.session_state.driver_info = {}
    
    # Login Section
    if not st.session_state.driver_logged_in:
        st.subheader("Driver Authentication")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
        
        with col2:
            username = st.text_input("Driver ID")
            password = st.text_input("Password", type="password")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Login", use_container_width=True):
                    # Simple authentication (in production, use Firebase Auth)
                    if username == "driver01" and password == "driver123":
                        st.session_state.driver_logged_in = True
                        st.session_state.driver_info = {
                            'name': 'Rajesh Kumar',
                            'id': 'DRV-1001',
                            'phone': '+91 98765 43210',
                            'vehicle': 'Electric Truck - E-Truck 101'
                        }
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Try: driver01 / driver123")
            
            with col_btn2:
                if st.button("Demo Login", use_container_width=True):
                    st.session_state.driver_logged_in = True
                    st.session_state.driver_info = {
                        'name': 'Demo Driver',
                        'id': 'DRV-DEMO',
                        'phone': '+91 98765 00000',
                        'vehicle': 'Demo Vehicle'
                    }
                    st.rerun()
        
        st.divider()
        st.info("""
        **Demo Credentials:**
        - Username: `driver01`
        - Password: `driver123`
        
        **Features Available After Login:**
        - View assigned routes
        - Emergency dispatch system
        - Navigation integration
        - Performance tracking
        """)
        return
    
    # Driver Dashboard
    driver = st.session_state.driver_info
    
    st.success(f"Welcome, {driver['name']}! ({driver['id']})")
    
    # Logout Button
    if st.sidebar.button("Logout", type="secondary"):
        st.session_state.driver_logged_in = False
        st.rerun()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Driver Information Card
        st.subheader("👤 Driver Information")
        st.markdown(f"""
        <div class="driver-card">
            <h4>{driver['name']}</h4>
            <p><strong>Driver ID:</strong> {driver['id']}</p>
            <p><strong>Contact:</strong> {driver['phone']}</p>
            <p><strong>Vehicle:</strong> {driver['vehicle']}</p>
            <p><strong>Current Status:</strong> <span class="status-normal">On Duty</span></p>
            <p><strong>Completed Pickups Today:</strong> 8</p>
            <p><strong>Performance Score:</strong> 94/100</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Today's Route
        st.subheader("🗺️ Today's Assigned Route")
        
        route_schedule = [
            {"time": "09:00 AM", "bin": "Bin_101", "location": "Main Entrance", "status": "✅ Completed"},
            {"time": "09:30 AM", "bin": "Bin_103", "location": "Parking Lot", "status": "✅ Completed"},
            {"time": "10:15 AM", "bin": "Bin_105", "location": "Cafeteria", "status": "🔄 In Progress"},
            {"time": "11:00 AM", "bin": "Bin_107", "location": "Admin Block", "status": "⏳ Pending"},
            {"time": "11:45 AM", "bin": "Bin_109", "location": "Library", "status": "⏳ Pending"},
            {"time": "12:30 PM", "bin": "Bin_111", "location": "Sports Complex", "status": "⏳ Pending"},
        ]
        
        for task in route_schedule:
            col_time, col_details, col_status = st.columns([1, 2, 1])
            with col_time:
                st.write(f"**{task['time']}**")
            with col_details:
                st.write(f"{task['bin']} - {task['location']}")
            with col_status:
                st.write(task['status'])
        
        # Navigation Controls
        st.subheader("🧭 Navigation")
        current_bin = "Bin_105"
        current_location = "Vasant Naik College Cafeteria"
        
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("📍 Open Google Maps", use_container_width=True):
                maps_url = f"https://www.google.com/maps/dir/?api=1&destination={Parbhani_COORDS['lat']},{Parbhani_COORDS['lon']}"
                st.markdown(f'<a href="{maps_url}" target="_blank">Open in Google Maps</a>', unsafe_allow_html=True)
        
        with col_nav2:
            if st.button("✅ Mark Bin as Collected", use_container_width=True):
                st.success(f"{current_bin} marked as collected! +10 Green Points")
    
    with col2:
        # Emergency Alerts
        st.subheader("🚨 Emergency Dispatch")
        
        # Simulate emergency bin
        if st.button("Simulate Emergency Bin (85%+)", use_container_width=True):
            st.session_state.emergency_simulated = True
        
        if st.session_state.get('emergency_simulated', False):
            st.error("""
            ⚠️ **EMERGENCY DISPATCH ALERT!**
            
            **Bin A-01** is 92% full and requires immediate attention!
            
            **Details:**
            - Location: Main Entrance
            - Distance: 2.3 km from your location
            - Estimated Time: 8 minutes
            - Priority: HIGH
            """)
            
            col_emg1, col_emg2 = st.columns(2)
            with col_emg1:
                if st.button("🗺️ Navigate to Emergency", use_container_width=True):
                    st.info("Opening optimized route to emergency bin...")
                    # In real implementation, open maps with coordinates
            
            with col_emg2:
                if st.button("✅ Complete Emergency Pickup", use_container_width=True):
                    st.session_state.emergency_simulated = False
                    st.success("Emergency pickup completed! +50 Green Points awarded.")
                    st.balloons()
                    st.rerun()
        
        # Performance Metrics
        st.subheader("📊 Performance Dashboard")
        
        metrics = [
            ("Today's Efficiency", "94%", "+2% from yesterday"),
            ("Avg Collection Time", "12 mins", "-3 mins improvement"),
            ("Fuel Saved", "4.2 liters", "30% savings"),
            ("Green Points", "320", "+40 this week"),
            ("Customer Rating", "4.8/5.0", "48 positive reviews"),
            ("CO2 Reduced", "42 kg", "This month")
        ]
        
        for metric, value, delta in metrics:
            st.metric(metric, value, delta)
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("📱 Call Dispatch Center", use_container_width=True):
            st.info("Calling: +91 98765 43210")
        
        if st.button("🔄 Refresh Route", use_container_width=True):
            st.info("Route refreshed with latest optimizations")
    
    st.divider()
    
    # Collection History
    st.subheader("📅 Collection History")
    
    # Generate sample history
    dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
    history_data = []
    
    for date in dates:
        history_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Bins Collected': random.randint(8, 15),
            'Total Weight (kg)': random.randint(200, 400),
            'Distance (km)': round(random.uniform(25, 45), 1),
            'Efficiency (%)': random.randint(85, 98)
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)

# ==========================================
# 👥 CITIZEN ENGAGEMENT PORTAL
# ==========================================

def show_citizen_engagement():
    """Citizen portal with reporting and gamification"""
    st.title("👥 Citizen Engagement Portal")
    
    tab1, tab2, tab3 = st.tabs(["📸 Report Garbage", "🏆 Green Leaderboard", "💬 Eco Chatbot"])
    
    with tab1:
        st.subheader("Report Improper Waste Disposal")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Report Form
            with st.form("report_form", clear_on_submit=True):
                st.write("### 📝 Submit a Report")
                
                # User Information
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    name = st.text_input("Your Name")
                with col_info2:
                    email = st.text_input("Email (optional)")
                
                # Location
                location = st.selectbox("Location", [
                    "Main Road", "Park Area", "Market", 
                    "Residential Area", "School", "Hospital",
                    "Commercial Area", "Beach", "Other"
                ])
                
                if location == "Other":
                    location = st.text_input("Specify Location")
                
                # Waste Details
                col_waste1, col_waste2 = st.columns(2)
                with col_waste1:
                    waste_type = st.selectbox("Type of Waste", [
                        "Plastic", "Organic/Food", "Paper", 
                        "Metal", "Glass", "Electronic",
                        "Construction", "Hazardous", "Mixed"
                    ])
                with col_waste2:
                    severity = st.select_slider("Severity", 
                                               options=["Low", "Medium", "High", "Critical"])
                
                # Image Upload
                uploaded_file = st.file_uploader("Upload photo of waste", 
                                               type=['jpg', 'png', 'jpeg'],
                                               help="Clear photos help with AI verification")
                
                # Additional Details
                description = st.text_area("Additional Details (optional)", 
                                         placeholder="Describe the issue, any odors, animals attracted, etc.")
                
                # Submit Button
                submitted = st.form_submit_button("Submit Report for AI Verification", 
                                                use_container_width=True)
                
                if submitted:
                    if not name:
                        st.error("Please enter your name")
                    elif not uploaded_file:
                        st.error("Please upload a photo for verification")
                    else:
                        with st.spinner("🔍 AI is analyzing your report..."):
                            # Simulate AI processing
                            time.sleep(2)
                            
                            # AI Classification
                            ai_result = classify_garbage_image(uploaded_file)
                            
                            # Award points based on report
                            base_points = 10
                            severity_multiplier = {"Low": 1, "Medium": 1.5, "High": 2, "Critical": 3}
                            points = int(base_points * severity_multiplier[severity])
                            
                            # Save report (in production, save to Firebase)
                            report_data = {
                                'timestamp': datetime.now(),
                                'reporter': name,
                                'email': email,
                                'location': location,
                                'waste_type': waste_type,
                                'severity': severity,
                                'ai_classification': ai_result['primary_category'],
                                'ai_confidence': ai_result['confidence'],
                                'points_awarded': points,
                                'status': 'Verified',
                                'description': description
                            }
                            
                            st.success(f"""
                            ✅ **Report Submitted Successfully!**
                            
                            **AI Analysis Results:**
                            - 📸 **Image Classification:** {ai_result['primary_category']}
                            - 🎯 **Confidence:** {ai_result['confidence']:.1%}
                            - 🏅 **Green Points Earned:** {points}
                            
                            **Report Details:**
                            - 📍 Location: {location}
                            - 🗑️ Waste Type: {waste_type}
                            - ⚠️ Severity: {severity}
                            
                            Your report has been forwarded to municipal authorities for action.
                            """)
                            
                            # Show image preview
                            st.image(uploaded_file, caption="Uploaded Waste Photo", width=300)
                            
                            # Celebration for critical reports
                            if severity == "Critical":
                                st.balloons()
        
        with col2:
            # Live Citizen Activity Feed
            st.subheader("📡 Live Citizen Reports")
            
            # Sample recent reports
            recent_reports = [
                {"user": "Green Warrior", "action": "Reported Plastic Waste", "points": 25, "time": "2 min ago"},
                {"user": "Eco Hero", "action": "Cleaned Park Area", "points": 50, "time": "5 min ago"},
                {"user": "Clean Champ", "action": "Reported Overflowing Bin", "points": 30, "time": "10 min ago"},
                {"user": "Waste Warrior", "action": "Organized Cleanup", "points": 100, "time": "15 min ago"},
                {"user": "Recycle Master", "action": "Proper Segregation", "points": 15, "time": "20 min ago"},
            ]
            
            for report in recent_reports:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #f8fff8 0%, #f0fff0 100%); 
                          padding: 12px; border-radius: 10px; margin: 8px 0; 
                          border-left: 4px solid #00C853;">
                    <div style="display: flex; align-items: center;">
                        <div style="background: #00C853; color: white; width: 30px; height: 30px; 
                                  border-radius: 50%; display: flex; align-items: center; 
                                  justify-content: center; margin-right: 10px; font-weight: bold;">
                            👤
                        </div>
                        <div>
                            <strong>{report['user']}</strong><br>
                            <small>{report['action']}</small>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                        <span style="color: #00C853; font-weight: bold;">+{report['points']} pts</span>
                        <span style="color: #888; font-size: 0.8rem;">{report['time']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Monthly Statistics
            st.subheader("📊 Monthly Impact")
            
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Reports Submitted", "1,245", "+18%")
            with stats_col2:
                st.metric("Waste Collected", "2.8 tons", "+25%")
            
            # Quick Tip
            st.info("""
            💡 **Tip of the Day:**
            Always separate wet and dry waste. 
            Wet waste can be composted, while dry waste can often be recycled.
            """)
    
    with tab2:
        st.subheader("🏆 Green Points Leaderboard")
        
        # Get leaderboard data
        leaderboard_data = generate_leaderboard()
        
        # Top 3 with special styling
        st.markdown("### 🥇 Top Performers")
        
        top_cols = st.columns(3)
        for i in range(3):
            with top_cols[i]:
                user = leaderboard_data[i]
                medal_class = ["gold", "silver", "bronze"][i]
                st.markdown(f"""
                <div class="leaderboard-card {medal_class}">
                    <h2 style="text-align: center; margin: 0;">#{i+1}</h2>
                    <h3 style="text-align: center; margin: 10px 0;">{user['name']}</h3>
                    <h1 style="text-align: center; margin: 10px 0; font-size: 2.5rem;">{user['points']}</h1>
                    <p style="text-align: center; margin: 5px 0;">Green Points</p>
                    <p style="text-align: center; margin: 5px 0;">🏅 {user['level']}</p>
                    <p style="text-align: center; margin: 5px 0; font-size: 0.9rem;">{user['reports']} reports</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Full Leaderboard
        st.markdown("### 📋 Full Leaderboard")
        
        for i, user in enumerate(leaderboard_data[3:], start=4):
            col_rank, col_name, col_points, col_level = st.columns([1, 3, 2, 2])
            with col_rank:
                st.write(f"**#{i}**")
            with col_name:
                st.write(f"👤 {user['name']}")
            with col_points:
                st.write(f"🏅 {user['points']} points")
            with col_level:
                st.write(f"📊 {user['level']} • {user['reports']} reports")
        
        # How to Earn Points
        st.divider()
        st.subheader("🎯 How to Earn Green Points")
        
        point_system = [
            ("Submit verified waste report", "10-30 points"),
            ("Participate in community cleanup", "50 points"),
            ("Consistently segregate waste", "5 points/day"),
            ("Report overflowing bins", "15 points"),
            ("Refer friends to the program", "25 points each"),
            ("Complete monthly challenges", "100 points"),
        ]
        
        for action, points in point_system:
            col_action, col_points = st.columns([3, 1])
            col_action.write(f"• {action}")
            col_points.write(f"**{points}**")
    
    with tab3:
        st.subheader("💬 EcoBot - Your Waste Management Assistant")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Hello! I'm EcoBot, your AI assistant for all things waste management. How can I help you today?"}
            ]
        
        # Chat container
        chat_container = st.container(height=400, border=True)
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style="background: #e3f2fd; padding: 12px 16px; 
                              border-radius: 18px 18px 18px 4px; margin: 8px 0; 
                              max-width: 80%; float: right; clear: both;">
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #f5f5f5; padding: 12px 16px; 
                              border-radius: 18px 18px 4px 18px; margin: 8px 0; 
                              max-width: 80%; float: left; clear: both;">
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Quick questions buttons
        st.markdown("**Quick Questions:**")
        
        quick_questions = [
            "What's recyclable?",
            "Nearest bin location?",
            "How to compost?",
            "Report overflowing bin",
            "Earn green points",
            "Schedule bulk pickup"
        ]
        
        cols = st.columns(3)
        for i, question in enumerate(quick_questions):
            with cols[i % 3]:
                if st.button(question, key=f"quick_{i}"):
                    # Add user message
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question
                    })
                    
                    # Generate AI response
                    responses = {
                        "What's recyclable?": """
                        **Recyclable Materials:**
                        - Clean plastic bottles and containers (PET, HDPE)
                        - Paper and cardboard (dry and clean)
                        - Glass jars and bottles
                        - Aluminum and steel cans
                        - Clean metal containers
                        
                        **Not Recyclable:**
                        - Food-contaminated items
                        - Plastic bags and films
                        - Styrofoam
                        - Ceramics and Pyrex
                        """,
                        "Nearest bin location?": """
                        Based on your location in **Parbhani**, the nearest smart bins are:
                        
                        1. **Bin_101** - Main Entrance (150m away) - 45% full
                        2. **Bin_103** - Parking Lot (280m away) - 32% full
                        3. **Bin_105** - Cafeteria (350m away) - 78% full ⚠️
                        
                        Use the Real-Time Monitoring dashboard for live updates!
                        """,
                        "How to compost?": """
                        **Home Composting Guide:**
                        
                        1. **Collect:** Vegetable scraps, fruit peels, coffee grounds, eggshells
                        2. **Avoid:** Meat, dairy, oils, diseased plants
                        3. **Layering:** Alternate green (nitrogen) and brown (carbon) materials
                        4. **Maintain:** Keep moist like a wrung-out sponge
                        5. **Turn:** Mix pile every 1-2 weeks for aeration
                        6. **Harvest:** Ready in 2-6 months
                        
                        Reduces landfill waste by 30%!
                        """,
                        "Report overflowing bin": """
                        **To report an overflowing bin:**
                        
                        1. Open the **Citizen Portal**
                        2. Click **"Report Garbage"** tab
                        3. Fill in location details
                        4. Upload a clear photo
                        5. Submit for AI verification
                        
                        Earn 10-30 Green Points per verified report!
                        """,
                        "Earn green points": """
                        **Ways to Earn Green Points:**
                        
                        🏅 **Daily Actions:**
                        - Report waste: 10-30 points
                        - Proper segregation: 5 points/day
                        
                        🏅 **Weekly Activities:**
                        - Community cleanup: 50 points
                        - Educational posts: 20 points
                        
                        🏅 **Monthly Challenges:**
                        - Zero waste week: 100 points
                        - Recycling drive: 150 points
                        
                        Top performers get featured on the leaderboard!
                        """,
                        "Schedule bulk pickup": """
                        **Bulk Waste Pickup Schedule:**
                        
                        **For Residents:**
                        - Call: 1800-XXX-XXX
                        - Online: www.parbhanimunicipal.gov.in
                        - WhatsApp: +91 XXXXX XXXXX
                        
                        **Accepted Items:**
                        - Furniture, mattresses
                        - Appliances, electronics
                        - Garden waste
                        - Construction debris (limited)
                        
                        Schedule 48 hours in advance.
                        """
                    }
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": responses.get(question, "I can help you with that! Please ask your question in detail.")
                    })
                    st.rerun()
        
        # User input
        user_input = st.text_input("Ask EcoBot anything about waste management:", 
                                 key="user_input",
                                 placeholder="Type your question here...")
        
        col_send, col_clear = st.columns([3, 1])
        with col_send:
            if st.button("Send", use_container_width=True):
                if user_input:
                    # Add user message
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Simulate AI response
                    ai_response = f"I understand you're asking about waste management. For specific questions about recycling, composting, or reporting issues, I recommend checking the relevant sections in the dashboard. Would you like me to help you with any particular aspect?"
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": ai_response
                    })
                    st.rerun()
        
        with col_clear:
            if st.button("Clear Chat", type="secondary", use_container_width=True):
                st.session_state.chat_history = [
                    {"role": "assistant", "content": "Hello! I'm EcoBot, your AI assistant for all things waste management. How can I help you today?"}
                ]
                st.rerun()

# ==========================================
# 📈 ANALYTICS & PREDICTIONS
# ==========================================
def show_analytics():
    st.title("📊 Data & Financials")

    tab1, tab2, tab3 = st.tabs([
        "📈 Exploratory Data Analysis (EDA)",
        "🧠 Predictive AI",
        "💎 Comprehensive Impact Model"
    ])

    # =====================================================
    # LOAD DATA (AUTO + UPLOAD FALLBACK)
    # =====================================================
    df = None

    try:
        df = pd.read_csv("smart_bin_historical_data.csv")
        st.success("✅ Historical Data Loaded Automatically")
    except FileNotFoundError:
        st.warning("⚠️ Historical data not found. Please upload the CSV file.")
        uploaded = st.file_uploader(
            "Upload smart_bin_historical_data.csv",
            type="csv"
        )
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.success("✅ File uploaded successfully")

    # =====================================================
    # TAB 1 — EDA
    # =====================================================
    with tab1:
        st.markdown("### 📈 Historical Patterns")

        if df is None:
            st.info("Upload data to view analytics.")
        else:
            col1, col2 = st.columns(2)

            # -------- Fill by Hour --------
            with col1:
                st.subheader("Fill by Hour")

                if {'hour_of_day', 'bin_fill_percent'}.issubset(df.columns):
                    hourly = (
                        df.groupby('hour_of_day')['bin_fill_percent']
                        .mean()
                        .reset_index()
                    )

                    fig1 = px.line(
                        hourly,
                        x='hour_of_day',
                        y='bin_fill_percent',
                        title='Hourly Fill Pattern',
                        labels={
                            'hour_of_day': 'Hour of Day',
                            'bin_fill_percent': 'Average Fill (%)'
                        }
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.warning("Required columns missing for hourly analysis.")

            # -------- Fill by Day --------
            with col2:
                st.subheader("Fill by Day")

                if {'day_of_week', 'bin_fill_percent'}.issubset(df.columns):
                    daily = (
                        df.groupby('day_of_week')['bin_fill_percent']
                        .mean()
                        .reset_index()
                    )

                    day_order = [
                        'Monday', 'Tuesday', 'Wednesday',
                        'Thursday', 'Friday', 'Saturday', 'Sunday'
                    ]

                    daily['day_of_week'] = pd.Categorical(
                        daily['day_of_week'],
                        categories=day_order,
                        ordered=True
                    )
                    daily = daily.sort_values('day_of_week')

                    fig2 = px.bar(
                        daily,
                        x='day_of_week',
                        y='bin_fill_percent',
                        title='Weekly Fill Pattern',
                        labels={
                            'day_of_week': 'Day',
                            'bin_fill_percent': 'Average Fill (%)'
                        }
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("Required columns missing for daily analysis.")

   
    # =====================================================



    
    # =====================================================
    with tab2:
        st.markdown("### 🧠 AI Forecast Training")
    
        if df is None:
            st.info("Upload data to train the model.")
        else:
            if st.button("Train Random Forest Model"):
                with st.spinner("Training Random Forest model..."):
                    try:
                        # -----------------------------
                        # Feature selection
                        # -----------------------------
                        feature_cols = [
                            'hour_of_day',
                            'day_of_week',
                            'ward',
                            'area_type',
                            'time_since_last_pickup'
                        ]
                        target_col = 'bin_fill_percent'
    
                        model_df = df[feature_cols + [target_col]].dropna()
    
                        if len(model_df) < 20:
                            st.warning("Not enough data to train the model.")
                        else:
                            # -----------------------------
                            # Encode categorical variables
                            # -----------------------------
                            model_df = pd.get_dummies(
                                model_df,
                                columns=['day_of_week', 'ward', 'area_type'],
                                drop_first=True
                            )
    
                            X = model_df.drop(target_col, axis=1)
                            y = model_df[target_col]
    
                            # -----------------------------
                            # Train / Test split
                            # -----------------------------
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
    
                            # -----------------------------
                            # Train model
                            # -----------------------------
                            model = RandomForestRegressor(
                                n_estimators=100,
                                random_state=42,
                                n_jobs=-1
                            )
                            model.fit(X_train, y_train)
    
                            # -----------------------------
                            # Evaluation
                            # -----------------------------
                            predictions = model.predict(X_test)
                            mae = mean_absolute_error(y_test, predictions)
                            r2 = r2_score(y_test, predictions)
    
                            st.success("✅ Model Trained Successfully")
    
                            c1, c2 = st.columns(2)
                            c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}%")
                            c2.metric("R² Score", f"{r2:.2f}")
    
                            # -----------------------------
                            # Prediction vs Actual plot
                            # -----------------------------
                            st.subheader("Prediction vs Actual")
    
                            plot_df = pd.DataFrame({
                                "Actual Fill (%)": y_test,
                                "Predicted Fill (%)": predictions
                            })
    
                            fig = px.scatter(
                                plot_df,
                                x="Actual Fill (%)",
                                y="Predicted Fill (%)",
                                title="Actual vs Predicted Bin Fill Levels",
                                trendline="ols"
                            )
    
                            fig.add_shape(
                                type="line",
                                x0=0, y0=0, x1=100, y1=100,
                                line=dict(color="red", dash="dash")
                            )
    
                            st.plotly_chart(fig, use_container_width=True)
    
                    except Exception as e:
                        st.error(f"Training failed: {e}")
        
    # TAB 3 — IMPACT MODEL
    # =====================================================
    with tab3:
        st.markdown("### 💎 Comprehensive Impact Model")

        st.info(
            "This section can be extended with cost savings, "
            "fuel reduction, and sustainability KPIs."
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Estimated Fuel Savings", "32%")
        k2.metric("Operational Cost Reduction", "₹1.2L / year")
        k3.metric("Overflow Reduction", "41%")

    
    
    
    
    
# =====================================================
#
# =====================================================



 
# 💰 FINANCIAL ROI MODEL
# ==========================================

def show_financial_model():
    """Financial ROI analysis page"""
    st.title("💰 Financial Impact & ROI Analysis")
    
    st.markdown("### The 360° Value Proposition")
    st.write("""
    This advanced model evaluates the project's viability across four dimensions: 
    **Operational Savings**, **Revenue Generation**, **Strategic Cost Avoidance**, and **Environmental Monetization**.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Simulation Parameters")
    
    # Toggle for EV Fleet
    is_ev = st.sidebar.checkbox("⚡ Activate Electric Vehicle (EV) Fleet Mode", value=True)
    
    # CAPEX Section
    st.sidebar.subheader("1. CAPEX (Initial Investment)")
    num_trucks = st.sidebar.number_input("Fleet Size (Trucks)", value=5, min_value=1)
    hardware_cost_per_bin = st.sidebar.number_input("Hardware Cost/Bin (₹)", value=1500)
    total_bins = st.sidebar.number_input("Total Smart Bins", value=100)
    software_dev_cost = st.sidebar.number_input("Software/Cloud Setup Cost (₹)", value=50000)
    
    # OPEX Section
    st.sidebar.subheader("2. OPEX (Operational)")
    driver_wage = st.sidebar.slider("Staff Hourly Wage (₹)", 100, 500, 200)
    
    # Dynamic Fuel/Energy Inputs
    if is_ev:
        st.sidebar.markdown("--- **⚡ EV Settings** ---")
        fuel_price = st.sidebar.number_input("Electricity Cost (₹/kWh)", value=10.0)
        truck_efficiency = st.sidebar.number_input("EV Efficiency (km/kWh)", value=1.5)
        co2_factor = 0.82  # kg CO2 per kWh (Grid Average)
        fuel_label = "Electricity"
        fuel_unit = "kWh"
    else:
        st.sidebar.markdown("--- **⛽ Diesel Settings** ---")
        fuel_price = st.sidebar.number_input("Diesel Price (₹/Liter)", value=104.0)
        truck_efficiency = st.sidebar.number_input("Truck Mileage (km/L)", value=4.0)
        co2_factor = 2.68  # kg CO2 per Liter
        fuel_label = "Fuel"
        fuel_unit = "L"
    
    # Additional Costs
    maintenance_per_km = st.sidebar.number_input("Vehicle Maintenance (₹/km)", value=5.0)
    cloud_cost_per_bin = st.sidebar.number_input("Cloud/Data Cost per Bin/Month (₹)", value=20)
    
    # Revenue & Strategic Value
    st.sidebar.subheader("3. Revenue & Strategic Value")
    
    recyclable_value_per_kg = st.sidebar.number_input("Avg. Recyclable Value (₹/kg)", value=15.0)
    recycling_rate_increase = st.sidebar.slider("Recycling Efficiency Boost (%)", 0, 50, 20)
    daily_waste_collected_kg = st.sidebar.number_input("Total Daily Waste (kg)", value=2000.0)
    
    penalty_per_overflow = st.sidebar.number_input("Fine per Overflowing Bin (₹)", value=500)
    overflows_prevented_month = st.sidebar.slider("Overflows Prevented/Month", 0, 100, 25)
    
    carbon_credit_price = st.sidebar.number_input("Carbon Credit Price (₹/Ton CO2)", value=1500.0)
    
    # Logistics Efficiency
    st.sidebar.subheader("4. Logistics Efficiency")
    
    st.sidebar.markdown("**🔴 Traditional (Fixed Route)**")
    dist_old = st.sidebar.number_input("Daily Distance Fixed (km)", value=60.0)
    trips_old = st.sidebar.slider("Trips/Month (Fixed)", 15, 30, 30)
    hours_old = st.sidebar.number_input("Hours/Trip (Fixed)", value=7.0)
    
    st.sidebar.markdown("**🟢 Smart (Optimized Route)**")
    dist_new = st.sidebar.number_input("Daily Distance Smart (km)", value=40.0)
    trips_new = st.sidebar.slider("Trips/Month (Smart)", 15, 30, 24)
    hours_new = st.sidebar.number_input("Hours/Trip (Smart)", value=5.0)
    
    # Prepare parameters for calculation
    params = {
        'hardware_cost_per_bin': hardware_cost_per_bin,
        'total_bins': total_bins,
        'software_dev_cost': software_dev_cost,
        'num_trucks': num_trucks,
        'driver_wage': driver_wage,
        'fuel_price': fuel_price,
        'truck_efficiency': truck_efficiency,
        'co2_factor': co2_factor,
        'maintenance_per_km': maintenance_per_km,
        'cloud_cost_per_bin': cloud_cost_per_bin,
        'recyclable_value_per_kg': recyclable_value_per_kg,
        'recycling_rate_increase': recycling_rate_increase,
        'daily_waste_collected_kg': daily_waste_collected_kg,
        'penalty_per_overflow': penalty_per_overflow,
        'overflows_prevented_month': overflows_prevented_month,
        'carbon_credit_price': carbon_credit_price,
        'dist_old': dist_old,
        'trips_old': trips_old,
        'hours_old': hours_old,
        'dist_new': dist_new,
        'trips_new': trips_new,
        'hours_new': hours_new,
    }
    
    # Calculate ROI
    roi_results = calculate_financial_roi(params)
    
    # Display Results
    st.markdown("### 📊 Monthly Financial Snapshot")
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Net Monthly Benefit", f"₹{roi_results['total_monthly_benefit']:,.0f}")
    with k2:
        st.metric("Direct OPEX Savings", f"₹{roi_results['opex_savings']:,.0f}")
    with k3:
        st.metric("Revenue & Avoidance", f"₹{roi_results['revenue_gain'] + roi_results['penalty_savings']:,.0f}")
    with k4:
        st.metric("ROI Break-even", f"{roi_results['months_breakeven']:.1f} Months")
    
    st.divider()
    
    # Detailed Breakdown
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Waterfall Chart
        waterfall_data = pd.DataFrame({
            "Source": ["Operational Savings", "Recycling Revenue", "Avoided Penalties", "Carbon Credits"],
            "Amount (₹)": [
                roi_results['opex_savings'],
                roi_results['revenue_gain'],
                roi_results['penalty_savings'],
                roi_results['carbon_credit_revenue']
            ]
        })
        
        fig_waterfall = px.bar(waterfall_data, x="Source", y="Amount (₹)", 
                              color="Source", text_auto='.2s',
                              title="Monthly Value Components",
                              color_discrete_sequence=['#00C853', '#1DE9B6', '#64DD17', '#AEEA00'])
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    with col2:
        # Environmental Impact
        st.subheader("🌍 Environmental Impact")
        st.metric("CO2 Prevented", f"{roi_results['co2_saved_tons']*1000:,.0f} kg")
        st.metric("Carbon Credit Value", f"₹{roi_results['carbon_credit_revenue']:,.0f}")
        
        # Equivalent metrics
        trees_equivalent = int(roi_results['co2_saved_tons'] * 1000 / 20)
        st.info(f"""
        **Equivalent to:**
        - Planting **{trees_equivalent} trees**
        - Removing **{int(roi_results['co2_saved_tons']*1000/2000)} cars** from roads
        - Saving **{int(roi_results['co2_saved_tons']*1000/8)} liters** of gasoline
        """)
    
    # 3-Year Financial Projection
    st.subheader("📈 3-Year Financial Projection")
    
    years = 3
    months = list(range(1, years * 12 + 1))
    
    cumulative_savings = []
    cumulative_costs = [roi_results['capex']] * len(months)
    net_cash_flow = []
    
    for i, month in enumerate(months):
        savings_month = roi_results['total_monthly_benefit'] * month
        cumulative_savings.append(savings_month)
        net_cash_flow.append(savings_month - roi_results['capex'])
    
    # Create DataFrame for plotting
    projection_df = pd.DataFrame({
        'Month': months,
        'Cumulative Savings': cumulative_savings,
        'Cumulative Costs': cumulative_costs,
        'Net Cash Flow': net_cash_flow
    })
    
    fig_projection = px.line(projection_df, x='Month', y=['Cumulative Savings', 'Cumulative Costs', 'Net Cash Flow'],
                           title='3-Year Financial Projection',
                           labels={'value': 'Amount (₹)', 'variable': 'Metric'})
    
    # Add break-even point
    if roi_results['months_breakeven'] > 0:
        fig_projection.add_vline(x=roi_results['months_breakeven'], line_dash="dash", 
                               line_color="green", annotation_text=f"Break-even: Month {int(roi_results['months_breakeven'])}")
    
    fig_projection.add_hline(y=0, line_dash="dot", line_color="gray")
    
    st.plotly_chart(fig_projection, use_container_width=True)
    
    # Investment Calculator
    st.divider()
    st.subheader("📱 Investment Calculator")
    
    calc_col1, calc_col2, calc_col3 = st.columns(3)
    
    with calc_col1:
        bins_count = st.number_input("Number of Bins", 10, 5000, 100, key="calc_bins")
    with calc_col2:
        area_size = st.selectbox("Area Coverage", 
                                ["Small (1-5 km²)", "Medium (5-20 km²)", "Large (20+ km²)"],
                                key="calc_area")
    with calc_col3:
        deployment = st.selectbox("Deployment Type", 
                                 ["Basic", "Standard", "Enterprise"],
                                 key="calc_deploy")
    
    # Calculate based on inputs
    base_cost = bins_count * 1800  # ₹1800 per bin
    
    if area_size == "Medium (5-20 km²)":
        base_cost *= 1.3
    elif area_size == "Large (20+ km²)":
        base_cost *= 1.6
    
    if deployment == "Standard":
        base_cost *= 1.25
    elif deployment == "Enterprise":
        base_cost *= 1.5
    
    monthly_saving = base_cost * 0.18  # 18% monthly savings
    payback_months = base_cost / monthly_saving if monthly_saving > 0 else 0
    
    st.success(f"""
    **Investment Summary:**
    
    - **Initial Investment:** ₹{base_cost:,.0f}
    - **Monthly Savings:** ₹{monthly_saving:,.0f}
    - **Payback Period:** {payback_months:.1f} months
    - **3-Year ROI:** {(monthly_saving * 36 - base_cost) / base_cost * 100:.0f}%
    - **Annual Savings:** ₹{monthly_saving * 12:,.0f}
    """)
    
    # Final Verdict
    st.divider()
    st.markdown("### 🎯 Final Verdict")
    
    st.success(f"""
    This project is not just a cost-saver; it is a **revenue generator**. 
    
    By integrating **Recycling Revenue** (₹{roi_results['revenue_gain']:,.0f}/month) and **Penalty Avoidance** (₹{roi_results['penalty_savings']:,.0f}/month) 
    with standard operational savings, the system pays for its hardware in **{roi_results['months_breakeven']:.1f} months**, 
    creating a sustainable profit model for the municipality.
    
    **Key Benefits:**
    1. **30% reduction** in operational costs
    2. **25% increase** in recycling revenue
    3. **40% fewer** overflow incidents
    4. **Positive ROI within 8-12 months**
    5. **Significant environmental impact** with CO2 reduction
    """)

# ==========================================
# 🔍 SYSTEM FEATURES PAGE
# ==========================================

def show_system_features():
    """System features and technology page"""
    st.title("🔍 System Features & Technology")
    
    st.markdown("""
    ### Intelligent System Components
    
    Our IoT solution combines cutting-edge hardware and cloud analytics to deliver 
    actionable insights for smarter waste management.
    """)
    
    # Features in a grid
    features = [
        {
            "icon": "🔌",
            "title": "ESP32 Microcontroller",
            "description": "Powerful, energy-efficient processor that manages sensor data collection and cloud communication seamlessly.",
            "benefits": ["Low power consumption", "WiFi & Bluetooth enabled", "Real-time processing"]
        },
        {
            "icon": "📡",
            "title": "Ultrasonic Sensors",
            "description": "Precisely measure bin fill levels in real-time, providing accurate data for collection scheduling.",
            "benefits": ["95% accuracy", "Non-contact measurement", "Works in all weather"]
        },
        {
            "icon": "⚖️",
            "title": "Load Cell Sensors",
            "description": "Track the weight of waste in each bin, providing insights into waste composition and density.",
            "benefits": ["Weight tracking", "Density analysis", "Theft prevention"]
        },
        {
            "icon": "☁️",
            "title": "Arduino IoT Cloud",
            "description": "Secure cloud infrastructure that stores, processes, and visualizes all sensor data in real-time.",
            "benefits": ["Secure data storage", "Real-time dashboards", "Scalable architecture"]
        },
        {
            "icon": "🤖",
            "title": "Predictive Models",
            "description": "Machine learning algorithms forecast fill times based on historical patterns and seasonal trends.",
            "benefits": ["24-48 hour predictions", "95% accuracy", "Adaptive learning"]
        },
        {
            "icon": "🗺️",
            "title": "Route Optimization",
            "description": "Google OR-Tools CVRP algorithm calculates the most efficient collection routes to save time and fuel.",
            "benefits": ["30% fuel savings", "Reduced travel time", "Dynamic routing"]
        },
    ]
    
    # Display features in a grid
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size: 2.5rem; margin-bottom: 15px;">{feature['icon']}</div>
                <h4 style="margin: 10px 0; color: #333;">{feature['title']}</h4>
                <p style="color: #666; margin-bottom: 15px; line-height: 1.5;">{feature['description']}</p>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 8px;">
                    <strong style="color: #00C853;">Key Benefits:</strong>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        {''.join([f'<li style="margin: 3px 0;">{benefit}</li>' for benefit in feature['benefits']])}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Deployment Scenarios
    st.subheader("🏙️ Built for Real-World Environments")
    
    st.markdown("""
    Our scalable, cost-effective solution adapts to various deployment scenarios, 
    delivering proven results across residential, institutional, and municipal settings.
    """)
    
    scenarios_cols = st.columns(3)
    
    scenarios = [
        {
            "title": "Residential Societies",
            "icon": "🏘️",
            "description": "Keep apartment complexes clean with automated monitoring and timely collection.",
            "metrics": ["90% satisfaction", "40% cost reduction", "Zero overflow"]
        },
        {
            "title": "Commercial Areas",
            "icon": "🏢",
            "description": "Optimize waste management for shopping malls, offices, and business districts.",
            "metrics": ["Efficient collection", "Clean environment", "Reduced complaints"]
        },
        {
            "title": "Municipal Systems",
            "icon": "🏛️",
            "description": "City-wide implementation for comprehensive waste management optimization.",
            "metrics": ["30% fuel savings", "25% labor savings", "Cleaner city"]
        },
    ]
    
    for i, scenario in enumerate(scenarios):
        with scenarios_cols[i]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                      padding: 25px; border-radius: 15px; height: 100%;">
                <div style="font-size: 3rem; text-align: center; margin-bottom: 15px;">
                    {scenario['icon']}
                </div>
                <h4 style="text-align: center; margin: 10px 0; color: #333;">
                    {scenario['title']}
                </h4>
                <p style="text-align: center; color: #666; margin-bottom: 20px;">
                    {scenario['description']}
                </p>
                <div style="background: white; padding: 15px; border-radius: 10px;">
                    {''.join([f'<div style="padding: 5px 0; border-bottom: 1px solid #eee;">✓ {metric}</div>' for metric in scenario['metrics']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Technology Stack
    st.subheader("🛠️ Technology Stack")
    
    tech_stack = {
        "Hardware": ["ESP32 Microcontroller", "HC-SR04 Ultrasonic Sensors", "HX711 Load Cells", "SIM800L GSM Module"],
        "Cloud & IoT": ["Arduino IoT Cloud", "Firebase Realtime Database", "Google Cloud Platform", "REST APIs"],
        "AI/ML": ["Scikit-learn", "Random Forest", "TensorFlow", "Hugging Face"],
        "Frontend": ["Streamlit", "Plotly", "Folium", "HTML/CSS"],
        "Optimization": ["Google OR-Tools", "CVRP Algorithm", "Dijkstra's Algorithm", "Route Optimization"],
    }
    
    for category, technologies in tech_stack.items():
        st.markdown(f"**{category}:**")
        tech_cols = st.columns(len(technologies))
        for i, tech in enumerate(technologies):
            with tech_cols[i]:
                st.markdown(f"""
                <div style="background: white; padding: 10px; border-radius: 8px; 
                          text-align: center; border: 1px solid #eee; margin: 5px;">
                    {tech}
                </div>
                """, unsafe_allow_html=True)
        st.write("")

# ==========================================
# 🎮 MAIN APPLICATION
# ==========================================

def main():
    """Main application controller"""
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    if 'driver_logged_in' not in st.session_state:
        st.session_state.driver_logged_in = False
    if 'emergency_simulated' not in st.session_state:
        st.session_state.emergency_simulated = False
    if 'dispatch_logs' not in st.session_state:
        st.session_state.dispatch_logs = []
    
    # Sidebar Navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063812.png", width=80)
        st.title("Smart Waste Monitoring")
        
        # Navigation Menu
        page_options = [
            "🏠 Home",
            "📊 Real-Time Monitoring",
            "🔍 System Features",
            "📈 Analytics & Predictions",
            "💰 Financial Model",
            "🚚 Driver Portal",
            "👥 Citizen Engagement",
        ]
        
        selected_page = st.radio(
            "Navigate",
            page_options,
            index=page_options.index(st.session_state.page) if st.session_state.page in page_options else 0
        )
        
        # Update page state
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()
        
        st.divider()
        
        # Quick Stats
        bins = generate_realtime_bin_data()
        critical_bins = len([b for b in bins if b['status'] == 'Critical'])
        
        st.metric("Active Bins", len(bins))
        st.metric("Need Attention", critical_bins)
        st.metric("Avg Fill Level", f"{np.mean([b['fill_level'] for b in bins]):.1f}%")
        
        st.divider()
        
        # System Status
        st.subheader("System Status")
        
        status_cols = st.columns(2)
        with status_cols[0]:
            st.success("✅ IoT Sensors")
        with status_cols[1]:
            st.info("🔄 AI Models")
        
        status_cols2 = st.columns(2)
        with status_cols2[0]:
            st.success("✅ Database")
        with status_cols2[1]:
            st.success("✅ Cloud Sync")
        
        st.divider()
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 10px; color: #666;">
            <small>IoT Smart Waste Monitoring System</small><br>
            <small>© 2025 All rights reserved</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Page Routing
    if st.session_state.page == "🏠 Home":
        show_landing_page()
    elif st.session_state.page == "📊 Real-Time Monitoring":
        show_realtime_monitoring()
    elif st.session_state.page == "🔍 System Features":
        show_system_features()
    elif st.session_state.page == "📈 Analytics & Predictions":
        show_analytics()
    elif st.session_state.page == "💰 Financial Model":
        show_financial_model()
    elif st.session_state.page == "🚚 Driver Portal":
        show_driver_portal()
    elif st.session_state.page == "👥 Citizen Engagement":
        show_citizen_engagement()

# ==========================================
# 🚀 RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    main()
