import streamlit as st
import numpy as np
import joblib
import requests
import folium
from streamlit_folium import st_folium

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Flood Risk Prediction Dashboard",
    page_icon="🌊",
    layout="centered"
)

# ----------------------------------
# LOAD ML MODEL
# ----------------------------------
model = joblib.load("flood_xgb_model.pkl")

# ----------------------------------
# OPENSTREETMAP GEOCODING (FREE)
# ----------------------------------
def geocode_location(place):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    headers = {"User-Agent": "FloodRiskDashboard"}
    response = requests.get(url, params=params, headers=headers).json()
    if response:
        return float(response[0]["lat"]), float(response[0]["lon"])
    return None, None

# ----------------------------------
# EXPLAINABLE AI FUNCTION
# ----------------------------------
def explain_prediction(rainfall, elevation, slope, river):
    explanations = []

    if rainfall > 120:
        explanations.append("• Very high rainfall increases surface water accumulation.")
    elif rainfall > 80:
        explanations.append("• Moderate rainfall contributes to flood risk.")

    if elevation < 30:
        explanations.append("• Low-lying terrain is prone to flooding.")
    elif elevation < 50:
        explanations.append("• Relatively low elevation increases susceptibility.")

    if river == 1:
        explanations.append("• Proximity to a river raises overflow risk.")

    if slope < 1:
        explanations.append("• Flat terrain slows water drainage.")

    return explanations

# ----------------------------------
# ALERT & SAFETY RECOMMENDATIONS
# ----------------------------------
def safety_recommendations(risk_level):
    if risk_level == 0:
        return [
            "• Normal conditions detected.",
            "• Stay informed about weather updates.",
            "• No immediate action required."
        ]
    elif risk_level == 1:
        return [
            "• Avoid low-lying and flood-prone roads.",
            "• Secure valuables and important documents.",
            "• Monitor rainfall and river level alerts."
        ]
    else:
        return [
            "• High flood risk detected.",
            "• Prepare for evacuation if advised by authorities.",
            "• Avoid travel near rivers and waterlogged areas.",
            "• Follow official disaster management advisories."
        ]

# ----------------------------------
# SESSION STATE (PERSIST OUTPUT)
# ----------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.sim_prediction = None
    st.session_state.risk_label = None
    st.session_state.risk_color = None
    st.session_state.lat = None
    st.session_state.lon = None
    st.session_state.rainfall = None
    st.session_state.elevation = None
    st.session_state.slope = None
    st.session_state.river_val = None
    st.session_state.sim_rainfall = None

# ----------------------------------
# HEADER
# ----------------------------------
st.title("🌊 Flood Risk Prediction Dashboard")
st.markdown(
    "AI-based flood risk assessment using **satellite rainfall**, "
    "**terrain elevation**, and **river proximity**."
)

st.divider()

# ----------------------------------
# INPUT FEATURES
# ----------------------------------
rainfall = st.slider("Observed rainfall (last 7 days, mm)", 0, 300, 120)
elevation = st.slider("Elevation (meters)", 0, 500, 30)
slope = st.slider("Slope (degrees)", 0.0, 20.0, 1.0)
river = st.selectbox("Near a river?", ["No", "Yes"])

# ----------------------------------
# WHAT-IF RAINFALL SIMULATION
# ----------------------------------
st.subheader("🌧️ What-If Rainfall Simulation")
rainfall_delta = st.slider(
    "Simulate rainfall change (mm)",
    min_value=-50,
    max_value=100,
    value=0,
    help="Simulate increase or decrease in rainfall"
)

simulated_rainfall = max(0, rainfall + rainfall_delta)
st.caption(f"Scenario rainfall: **{simulated_rainfall} mm**")

# ----------------------------------
# LOCATION SEARCH
# ----------------------------------
location_query = st.text_input(
    "Search location (City, State, Country)",
    value="Guwahati, Assam, India"
)

if st.button("Find Location"):
    lat, lon = geocode_location(location_query)
    if lat is not None and lon is not None:
        st.session_state.lat = lat
        st.session_state.lon = lon
        st.success(f"📍 Location found: {lat:.4f}, {lon:.4f}")
    else:
        st.session_state.lat = None
        st.session_state.lon = None
        st.error("Location not found. Try a more specific name.")

st.divider()

# ----------------------------------
# PREDICT FLOOD RISK
# ----------------------------------
if st.button("Predict Flood Risk"):

    if st.session_state.lat is None or st.session_state.lon is None:
        st.error("Please search and confirm a location first.")
        st.stop()

    river_val = 1 if river == "Yes" else 0

    # Base prediction
    base_input = np.array([[rainfall, elevation, slope, river_val]])
    base_pred = model.predict(base_input)[0]

    # Scenario prediction
    sim_input = np.array([[simulated_rainfall, elevation, slope, river_val]])
    sim_pred = model.predict(sim_input)[0]

    # Store state
    st.session_state.prediction = base_pred
    st.session_state.sim_prediction = sim_pred
    st.session_state.sim_rainfall = simulated_rainfall
    st.session_state.rainfall = rainfall
    st.session_state.elevation = elevation
    st.session_state.slope = slope
    st.session_state.river_val = river_val

    def label_color(pred):
        if pred == 0:
            return "LOW FLOOD RISK", "green"
        elif pred == 1:
            return "MEDIUM FLOOD RISK", "orange"
        else:
            return "HIGH FLOOD RISK", "red"

    st.session_state.risk_label, st.session_state.risk_color = label_color(sim_pred)

# ----------------------------------
# DISPLAY RESULTS
# ----------------------------------
if (
    st.session_state.prediction is not None and
    st.session_state.lat is not None and
    st.session_state.lon is not None
):

    # Risk comparison
    st.subheader("📊 Flood Risk Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Current Conditions")
        if st.session_state.prediction == 0:
            st.success("🟢 LOW RISK")
        elif st.session_state.prediction == 1:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

    with col2:
        st.markdown("### Simulated Scenario")
        if st.session_state.sim_prediction == 0:
            st.success("🟢 LOW RISK")
        elif st.session_state.sim_prediction == 1:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

    # -------------------------------
    # MAP
    # -------------------------------
    st.subheader("📍 Flood Risk Location Map")

    m = folium.Map(
        location=[st.session_state.lat, st.session_state.lon],
        zoom_start=6,
        tiles="CartoDB dark_matter"
    )

    folium.CircleMarker(
        location=[st.session_state.lat, st.session_state.lon],
        radius=10,
        popup=st.session_state.risk_label,
        color=st.session_state.risk_color,
        fill=True,
        fill_color=st.session_state.risk_color,
        fill_opacity=0.85
    ).add_to(m)

    st_folium(m, width=700, height=450)

    # -------------------------------
    # EXPLAINABLE AI PANEL
    # -------------------------------
    st.subheader("🧠 Why this prediction?")
    explanations = explain_prediction(
        st.session_state.sim_rainfall,
        st.session_state.elevation,
        st.session_state.slope,
        st.session_state.river_val
    )
    for item in explanations:
        st.markdown(item)

    # -------------------------------
    # ALERTS & SAFETY
    # -------------------------------
    st.subheader("🚨 Alerts & Safety Recommendations")
    alerts = safety_recommendations(st.session_state.sim_prediction)
    for alert in alerts:
        st.markdown(alert)

# ----------------------------------
# FOOTER
# ----------------------------------
st.divider()
st.caption(
    "Satellite data processed with Google Earth Engine | "
    "ML: XGBoost | UI: Streamlit | Maps: OpenStreetMap"
)
