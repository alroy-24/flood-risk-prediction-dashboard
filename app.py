import streamlit as st
import numpy as np
import joblib
import requests
import folium
from streamlit_folium import st_folium
import google.generativeai as genai

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
# GEMINI SETUP
# -------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------------------------------
# SAFE GEOCODING
# -------------------------------------------------
def geocode_location(place):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1}
        headers = {"User-Agent": "FloodRiskDashboard"}
        r = requests.get(url, params=params, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None

# -------------------------------------------------
# RULE-BASED EXPLANATION
# -------------------------------------------------
def explain_prediction(rainfall, elevation, slope, river):
    exp = []
    if rainfall > 120:
        exp.append("• Very high rainfall significantly increases flood risk.")
    elif rainfall > 80:
        exp.append("• Moderate rainfall contributes to flood risk.")
    if elevation < 30:
        exp.append("• Low elevation makes the area flood-prone.")
    if slope < 1:
        exp.append("• Flat terrain slows natural drainage.")
    if river == 1:
        exp.append("• Proximity to a river increases overflow probability.")
    return exp

# -------------------------------------------------
# SAFETY RECOMMENDATIONS
# -------------------------------------------------
def safety_recommendations(level):
    if level == 0:
        return ["• No immediate flood threat detected."]
    elif level == 1:
        return [
            "• Avoid low-lying roads.",
            "• Monitor rainfall alerts."
        ]
    else:
        return [
            "• High flood risk detected.",
            "• Prepare for evacuation.",
            "• Follow disaster management advisories."
        ]

# -------------------------------------------------
# GEMINI EXPLANATION (QUOTA SAFE)
# -------------------------------------------------
def gemini_explanation(prediction, features):
    risk_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    prompt = f"""
Explain flood risk assessment in simple terms.

Risk Level: {risk_map[prediction]}
Rainfall: {features['rainfall']} mm
Elevation: {features['elevation']} m
Slope: {features['slope']}°
Near River: {features['river']}

Explain clearly for disaster preparedness.
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception:
        return "⚠️ Gemini explanation unavailable due to API quota limits."

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.sim_prediction = None
    st.session_state.lat = None
    st.session_state.lon = None
    st.session_state.inputs = {}
    st.session_state.gemini_generated = False
    st.session_state.gemini_text = ""

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
# 🌊 Flood Risk Prediction Dashboard
**AI-powered flood risk assessment using satellite-derived data and machine learning**
""")
st.caption("Built using Google Earth Engine, XGBoost, Gemini & Streamlit")
st.divider()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("📥 Input Parameters")

    rainfall = st.slider("Rainfall (last 7 days, mm)", 0, 300, 120)
    elevation = st.slider("Elevation (meters)", 0, 500, 30)
    slope = st.slider("Slope (degrees)", 0.0, 20.0, 1.0)
    river = st.selectbox("Near a river?", ["No", "Yes"])

    st.divider()
    st.subheader("🌧️ What-If Rainfall Simulation")
    rainfall_delta = st.slider("Rainfall change (mm)", -50, 100, 0)
    simulated_rainfall = max(0, rainfall + rainfall_delta)

    st.divider()
    st.subheader("📍 Location Search (Best effort)")
    location_query = st.text_input("City, State, Country", "Guwahati, Assam, India")

    if st.button("📍 Find Location"):
        lat, lon = geocode_location(location_query)
        if lat:
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.success("Location found")
        else:
            st.warning("Text search failed. Use manual coordinates below.")

    st.divider()
    st.subheader("📌 Manual Coordinates (Reliable)")
    manual_lat = st.number_input("Latitude", value=26.1445, format="%.6f")
    manual_lon = st.number_input("Longitude", value=91.7362, format="%.6f")

    if st.button("📌 Use Manual Coordinates"):
        st.session_state.lat = manual_lat
        st.session_state.lon = manual_lon
        st.success("Coordinates set")

    st.divider()
    predict_clicked = st.button("🚨 Predict Flood Risk")

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if predict_clicked:
    if st.session_state.lat is None:
        st.error("Please set a location first.")
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
        "river": river_val,
    }

    st.session_state.gemini_generated = False
    st.session_state.gemini_text = ""

# -------------------------------------------------
# RESULTS
# -------------------------------------------------
if st.session_state.prediction is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Conditions")
        st.success("LOW RISK" if st.session_state.prediction == 0 else
                   "MEDIUM RISK" if st.session_state.prediction == 1 else
                   "HIGH RISK")

    with col2:
        st.subheader("Simulated Scenario")
        st.success("LOW RISK" if st.session_state.sim_prediction == 0 else
                   "MEDIUM RISK" if st.session_state.sim_prediction == 1 else
                   "HIGH RISK")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Map", "📜 Rules Explanation", "🚨 Alerts", "✨ Gemini Explanation"]
    )

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
            fill=True
        ).add_to(m)
        st_folium(m, width=900, height=450)

    with tab2:
        for e in explain_prediction(**st.session_state.inputs):
            st.markdown(e)

    with tab3:
        for a in safety_recommendations(st.session_state.sim_prediction):
            st.markdown(a)

    with tab4:
        if not st.session_state.gemini_generated:
            if st.button("Generate Gemini explanation"):
                with st.spinner("Calling Gemini..."):
                    st.session_state.gemini_text = gemini_explanation(
                        st.session_state.sim_prediction,
                        st.session_state.inputs
                    )
                    st.session_state.gemini_generated = True

        if st.session_state.gemini_generated:
            st.markdown(st.session_state.gemini_text)
            st.caption("Gemini explanation is generated once per prediction to conserve API quota.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption("⚠️ Academic & demonstration use only • Cloud-safe deployment")
