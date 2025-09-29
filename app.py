#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import joblib
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
import folium
from streamlit_folium import st_folium
import openrouteservice

# --- Load model ---
model = joblib.load("uber_rf_model.pkl")
features = ['passenger_count', 'distance_km', 'pickup_hour', 'pickup_dow', 'is_weekend']

# --- Streamlit settings ---
st.set_page_config(page_title="Uber Fare Prediction", layout="centered")
st.title("Uber Fare Prediction with Real Route 🗺️")

# --- OpenRouteService client ---
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjZmYTExYzcyZjhlNTRkMWVhYTllMmZmMmJhMDU1YjVkIiwiaCI6Im11cm11cjY0In0="  # Replace with your key
client = openrouteservice.Client(key=ORS_API_KEY)

# --- Initialize session state ---
if "pickup" not in st.session_state:
    st.session_state.pickup = None
if "dropoff" not in st.session_state:
    st.session_state.dropoff = None

st.write("📍 Click once for **Pickup** and again for **Dropoff** on the map.")

# --- Create Folium map ---
m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)

# Add markers if already selected
if st.session_state.pickup:
    folium.Marker(location=st.session_state.pickup, popup="Pickup", icon=folium.Icon(color="green")).add_to(m)

if st.session_state.dropoff:
    folium.Marker(location=st.session_state.dropoff, popup="Dropoff", icon=folium.Icon(color="red")).add_to(m)

    # --- Draw real route using ORS ---
    try:
        coords = [
            (st.session_state.pickup[1], st.session_state.pickup[0]),  # ORS uses (lon, lat)
            (st.session_state.dropoff[1], st.session_state.dropoff[0])
        ]
        route = client.directions(coords, profile='driving-car', format='geojson')
        geometry = route['features'][0]['geometry']['coordinates']
        # Convert to (lat, lon)
        route_latlon = [(lat, lon) for lon, lat in geometry]
        folium.PolyLine(route_latlon, color="blue", weight=5, opacity=0.7).add_to(m)
        distance_km = route['features'][0]['properties']['segments'][0]['distance'] / 1000
    except Exception as e:
        st.warning(f"Route generation failed: {e}")
        distance_km = geodesic(st.session_state.pickup, st.session_state.dropoff).km
else:
    distance_km = 0

# --- Display map once ---
map_data = st_folium(m, width=700, height=500)

# --- Detect clicks ---
if map_data and map_data["last_clicked"]:
    lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    if st.session_state.pickup is None:
        st.session_state.pickup = (lat, lon)
        st.success(f"✅ Pickup selected: {st.session_state.pickup}")
    elif st.session_state.dropoff is None:
        st.session_state.dropoff = (lat, lon)
        st.success(f"✅ Dropoff selected: {st.session_state.dropoff}")
    else:
        st.warning("🔄 Resetting pickup & dropoff… Click again to choose new points.")
        st.session_state.pickup = (lat, lon)
        st.session_state.dropoff = None

# --- Passenger count ---
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1)

# --- Date & Time ---
pickup_datetime = datetime.now()
pickup_hour = pickup_datetime.hour
pickup_dow = pickup_datetime.weekday()
is_weekend = 1 if pickup_dow in [5, 6] else 0

# --- Prediction ---
if st.button("Predict Fare"):
    if st.session_state.pickup and st.session_state.dropoff:
        X = pd.DataFrame([[passenger_count, distance_km, pickup_hour, pickup_dow, is_weekend]], columns=features)
        predicted_fare = model.predict(X)[0]
        st.info(f"🛣️ Distance: {distance_km:.2f} km ")
        st.success(f"💵 Predicted Fare: ${predicted_fare:.2f}")
    else:
        st.error("⚠️ Please select both pickup and dropoff on the map.")


# In[ ]:




