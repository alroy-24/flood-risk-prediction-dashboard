import streamlit as st
import numpy as np
import joblib
import requests
import folium
from streamlit_folium import st_folium

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Flood Risk Prediction",
    page_icon="🌊",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load("flood_xgb_model.pkl")

# -------------------------------------------------
# SAFE GEOCODING (STREAMLIT CLOUD COMPATIBLE)
# -------------------------------------------------
def geocode_location(place):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1}
        headers = {"User-Agent": "FloodRiskDashboard"}

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=5
        )

        response.raise_for_status()
        data = response.json()

        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])

    except Exception:
        return None, None

# -------------------------------------------------
# EXPLAINABLE AI
# -------------------------------------------------
def explain_prediction(rainfall, elevation, slope, river):
    explanations = []

    if rainfall > 120:
        explanations.append("• Very high rainfall significantly increases flood risk.")
    elif rainfall > 80:
        explanations.append("• Moderate rainfall contributes to water accumulation.")

    if elevation < 30:
        explanations.append("• Low elevation makes the area flood-prone.")
    elif elevation < 50:
        explanations.append("• Relatively low elevation increases vulnerability.")

    if river == 1:
        explanations.append("• Proximity to a river increases overflow probability.")

    if slope < 1:
        explanations.append("• Flat terrain slows natural drainage.")

    return explanations

# -------------------------------------------------
# SAFETY RECOMMENDATIONS
# -------------------------------------------------
def safety_recommendations(level):
    if level == 0:
        return [
            "• No immediate flood threat detected.",
            "• Stay informed with weather updates."
        ]
    elif level == 1:
        return [
            "• Avoid low-lying and flood-prone roads.",
            "• Secure valuables and important documents.",
            "• Monitor rainfall and river level alerts."
        ]
    else:
        return [
            "• High flood risk detected.",
            "• Prepare for evacuation if advised.",
            "• Avoid travel near rivers and waterlogged areas.",
            "• Follow official disaster management advisories."
        ]

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.sim_prediction = None
    st.session_state.lat = None
    st.session_state.lon = None
    st.session_state.inputs = {}

# -------------------------------------------------
# HERO HEADER
# -------------------------------------------------
st.markdown("""
# 🌊 Flood Risk Prediction Dashboard
**AI-powered flood risk assessment using satellite-derived data and machine learning**
""")

st.caption("Built using Google Earth Engine, XGBoost & Streamlit")

st.divider()

# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------
with st.sidebar:
    st.header("📥 Input Parameters")

    rainfall = st.slider("Observed rainfall (last 7 days, mm)", 0, 300, 120)
    elevation = st.slider("Elevation (meters)", 0, 500, 30)
    slope = st.slider("Slope (degrees)", 0.0, 20.0, 1.0)
    river = st.selectbox("Near a river?", ["No", "Yes"])

    st.divider()

    st.subheader("🌧️ What-If Rainfall Simulation")
    rainfall_delta = st.slider(
        "Rainfall change (mm)",
        -50, 100, 0,
        help="Simulate increased or decreased rainfall"
    )

    simulated_rainfall = max(0, rainfall + rainfall_delta)
    st.caption(f"Scenario rainfall: **{simulated_rainfall} mm**")

    st.divider()

    st.subheader("📍 Location Search")
    location_query = st.text_input(
        "City, State, Country",
        value="Guwahati, Assam, India"
    )

    if st.button("📍 Find Location"):
        lat, lon = geocode_location(location_query)
        if lat is not None and lon is not None:
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.success("Location found")
        else:
            st.error("Location lookup failed. Try again later.")

    st.divider()

    predict_clicked = st.button("🚨 Predict Flood Risk")

# -------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------
if predict_clicked:

    if st.session_state.lat is None:
        st.error("Please find a location first.")
        st.stop()

    river_val = 1 if river == "Yes" else 0

    base_input = np.array([[rainfall, elevation, slope, river_val]])
    sim_input = np.array([[simulated_rainfall, elevation, slope, river_val]])

    st.session_state.prediction = model.predict(base_input)[0]
    st.session_state.sim_prediction = model.predict(sim_input)[0]

    st.session_state.inputs = {
        "rainfall": simulated_rainfall,
        "elevation": elevation,
        "slope": slope,
        "river": river_val
    }

# -------------------------------------------------
# RESULTS
# -------------------------------------------------
if st.session_state.prediction is not None:

    st.subheader("📊 Prediction Results")

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

    # -------------------------------------------------
    # TABS
    # -------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["🗺️ Map", "🧠 Explanation", "🚨 Alerts"])

    with tab1:
        m = folium.Map(
            location=[st.session_state.lat, st.session_state.lon],
            zoom_start=6,
            tiles="CartoDB dark_matter"
        )

        folium.CircleMarker(
            location=[st.session_state.lat, st.session_state.lon],
            radius=10,
            color="red",
            fill=True,
            fill_opacity=0.8
        ).add_to(m)

        st_folium(m, width=900, height=450)

    with tab2:
        st.subheader("Why this prediction?")
        for e in explain_prediction(
            st.session_state.inputs["rainfall"],
            st.session_state.inputs["elevation"],
            st.session_state.inputs["slope"],
            st.session_state.inputs["river"]
        ):
            st.markdown(e)

    with tab3:
        st.subheader("Safety Recommendations")
        for a in safety_recommendations(st.session_state.sim_prediction):
            st.markdown(a)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption(
    "⚠️ For academic and demonstration purposes only • "
    "Live deployment on Streamlit Cloud"
)
