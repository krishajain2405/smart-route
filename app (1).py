# --- MAIN NAVIGATION LOGIC ---
if page == "Home":
    st.title("Smart Waste Management Analytics Dashboard")
    st.write("Welcome! This dashboard provides a comprehensive overview of the Smart Bin project's data science components.")
    st.subheader("Project Overview")
    st.write("- **Live Data:** Real-time fill-level data sent to a cloud dashboard.")
    st.write("- **Historical Analysis:** Patterns analyzed to understand waste generation.")
    st.write("- **Predictive Modeling:** ML models forecasting bin overflow.")
    st.write("- **Multi-Trip Optimization:** AI slicing massive workloads into manageable driver trips.")

# --- THE FIXED ROUTE OPTIMIZATION BLOCK ---
elif page == "Route Optimization":
    st.title("🚛 AI Multi-Fleet Mission Control")

    # 1. Sidebar Mission Controls
    st.sidebar.header("🕹️ Dispatch Controls")
    selected_truck = st.sidebar.selectbox("Active Truck", list(GARAGES.keys()))
    threshold = st.sidebar.slider("Fill Threshold (%)", 0, 100, 75)
    
    # Simulation Time Slider (Fixes the data visibility issue)
    times = sorted(df['timestamp'].unique())
    default_idx = int(len(times) * 0.85)
    sim_time = st.sidebar.select_slider("Select Simulation Time", options=times, value=times[default_idx])
    
    # 2. Assignment & Filter Logic
    df_snap = df[df['timestamp'] == sim_time].copy()

    def assign_truck(row):
        loc = (row['lat'], row['lon'])
        dists = {name: ((loc[0]-c[0])**2 + (loc[1]-c[1])**2)**0.5 for name, c in GARAGES.items()}
        return min(dists, key=dists.get)

    df_snap['assigned_truck'] = df_snap.apply(assign_truck, axis=1)
    
    # Filter for full bins assigned specifically to the selected truck
    all_my_bins = df_snap[(df_snap['assigned_truck'] == selected_truck) & (df_snap['fill'] >= threshold)]
    all_my_bins = all_my_bins.sort_values('fill', ascending=False)

    # 3. Multi-Trip Pipeline Logic
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Trip Pipeline")
    bins_per_trip = 8 
    num_trips = (len(all_my_bins) // bins_per_trip) + (1 if len(all_my_bins) % bins_per_trip > 0 else 0)
    
    if num_trips > 0:
        trip_num = st.sidebar.selectbox(
            f"Select Trip (Total: {num_trips})", 
            range(1, num_trips + 1), 
            format_func=lambda x: f"Trip {x}"
        )
        current_bins = all_my_bins.iloc[(trip_num-1)*bins_per_trip : trip_num*bins_per_trip]
    else:
        current_bins = pd.DataFrame()

    # 4. Metrics & Solver
    c1, c2, c3 = st.columns(3)
    c1.metric("Vehicle ID", selected_truck)
    c2.metric("Total Pending Bins", len(all_my_bins))
    c3.metric("Current Trip Stops", len(current_bins))

    if not current_bins.empty:
        if st.button(f"Calculate Optimized Route for {selected_truck} - Trip {trip_num}"):
            with st.spinner("Finding most efficient route..."):
                current_bins['demand_liters'] = (current_bins['fill'] / 100) * current_bins['bin_capacity_liters']
                garage_loc = GARAGES[selected_truck]
                depot_row = pd.DataFrame([{'lat': garage_loc[0], 'lon': garage_loc[1], 'demand_liters': 0, 'bin_id': 'Garage'}], index=[0])
                route_data = pd.concat([depot_row, current_bins]).reset_index(drop=True)
                
                # --- OR-TOOLS SOLVER ENGINE ---
                data = {
                    'locations': list(zip(route_data['lat'], route_data['lon'])),
                    'demands': [int(d) for d in route_data['demand_liters']],
                    'vehicle_capacities': [30000], 'num_vehicles': 1, 'depot': 0
                }
                manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
                routing = pywrapcp.RoutingModel(manager)
                
                def distance_callback(from_idx, to_idx):
                    p1, p2 = data['locations'][manager.IndexToNode(from_idx)], data['locations'][manager.IndexToNode(to_idx)]
                    return int(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 * 100000)
                
                routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance_callback))
                search_params = pywrapcp.DefaultRoutingSearchParameters()
                search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
                solution = routing.SolveWithParameters(search_params)

                if solution:
                    index, route_coords = routing.Start(0), []
                    while not routing.IsEnd(index):
                        route_coords.append(data['locations'][manager.IndexToNode(index)])
                        index = solution.Value(routing.NextVar(index))
                    route_coords.append(data['locations'][manager.IndexToNode(index)])
                    route_coords.append(DEONAR_DUMPING) 

                    # 5. Map & QR
                    m = folium.Map(location=[19.0760, 72.8777], zoom_start=12, tiles="CartoDB positron")
                    folium.PolyLine(route_coords, color="blue", weight=5).add_to(m)
                    folium.Marker(garage_loc, icon=folium.Icon(color='red', icon='truck', prefix='fa')).add_to(m)
                    for _, r in current_bins.iterrows():
                        folium.Marker([r['lat'], r['lon']], icon=folium.Icon(color='blue', icon='trash', prefix='fa')).add_to(m)
                    st_folium(m, width=1100, height=500, key="optimized_map")
                    
                    st.subheader("📲 Driver Navigation")
                    url = f"https://www.google.com/maps/dir/?api=1&origin={garage_loc[0]},{garage_loc[1]}&destination={DEONAR_DUMPING[0]},{DEONAR_DUMPING[1]}"
                    st.image(qrcode.make(url).get_image(), width=200)
    else:
        st.info("Adjust the threshold or simulation time to see active missions.")
