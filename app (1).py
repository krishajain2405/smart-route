# --- Route Optimization Page (Multi-Trip Pipeline) ---
elif page == "Route Optimization":
    st.title("🚛 AI Multi-Fleet Mission Control")

    # Sidebar Controls
    st.sidebar.header("🕹️ Dispatch Controls")
    selected_truck = st.sidebar.selectbox("Active Truck", list(GARAGES.keys()))
    threshold = st.sidebar.slider("Fill Threshold (%)", 0, 100, 75)
    
    # Simulation Time Slider
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
    bins_per_trip = 8 
    num_trips = (len(all_my_bins) // bins_per_trip) + (1 if len(all_my_bins) % bins_per_trip > 0 else 0)
    
    if num_trips > 0:
        trip_num = st.sidebar.selectbox(f"Select Trip (Total: {num_trips})", range(1, num_trips + 1), format_func=lambda x: f"Trip {x}")
        current_bins = all_my_bins.iloc[(trip_num-1)*bins_per_trip : trip_num*bins_per_trip]
    else:
        current_bins = pd.DataFrame()
