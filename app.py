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
# PRE-GENERATED AI EXPLANATIONS (NO GEMINI CALLS)
# -------------------------------------------------
PREGENERATED_AI_EXPLANATIONS = {
    0: """
### 🟢 Low Flood Risk — AI Explanation

The model predicts a **low flood risk** for this location.

- Rainfall levels are within safe limits
- Elevation supports natural drainage
- Terrain slope allows water runoff
- No significant river overflow risk detected

💡 **Recommendation:** Stay informed with weather updates.
""",

    1: """
### 🟡 Medium Flood Risk — AI Explanation

The model predicts a **moderate flood risk**.

- Rainfall may cause surface water accumulation
- Low-to-moderate elevation increases vulnerability
- Nearby water bodies can overflow under heavy rain

⚠️ **Recommendation:** Avoid low-lying areas and monitor alerts.
""",

    2: """
### 🔴 High Flood Risk — AI Explanation

The model predicts a **high flood risk** scenario.

- Heavy rainfall significantly increases runoff
- Low elevation limits drainage capacity
- Flat terrain slows water movement
- River overflow is likely under current conditions

🚨 **Recommendation:** Prepare for evacuation and follow official advisories.
"""
}

# -------------------------------------------------
# SAFE GEOCODING USING OPENCAGE
# -------------------------------------------------
OPENCAGE_KEY = st.secrets["geocoding"]["OPENCAGE_KEY"]

def geocode_location(place):
    try:
        url = "https://api.opencagedata.com/geocode/v1/json"
        params = {
            "q": place,
            "key": OPENCAGE_KEY,
            "limit": 1,
            "no_annotations": 1,
        }
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()

        if data["results"]:
            lat = data["results"][0]["geometry"]["lat"]
            lon = data["results"][0]["geometry"]["lng"]
            return lat, lon
    except Exception:
        pass
    return None, None

# -------------------------------------------------
# RULE-BASED EXPLANATION
# -------------------------------------------------
def rule_explanation(rainfall, elevation, slope, river):
    exp = []
    if rainfall > 120:
        exp.append("• Heavy rainfall increases flood risk.")
    if elevation < 30:
        exp.append("• Low elevation makes the area flood-prone.")
    if slope < 1:
        exp.append("• Flat terrain slows water drainage.")
    if river == 1:
        exp.append("• Proximity to river increases overflow risk.")
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
# SESSION STATE
# -------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.sim_prediction = None
    st.session_state.lat = None
    st.session_state.lon = None
    st.session_state.inputs = {}

# -------------------------------------------------
# HEADER
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

    rainfall = st.slider("Rainfall (last 7 days, mm)", 0, 300, 120)
    elevation = st.slider("Elevation (meters)", 0, 500, 30)
    slope = st.slider("Slope (degrees)", 0.0, 20.0, 1.0)
    river = st.selectbox("Near a river?", ["No", "Yes"])

    st.divider()
    st.subheader("🌧️ What-If Rainfall Simulation")
    rainfall_delta = st.slider("Rainfall change (mm)", -50, 100, 0)
    simulated_rainfall = max(0, rainfall + rainfall_delta)

    st.divider()
    st.subheader("📍 Location Search")
    location_query = st.text_input("City, State, Country", "Guwahati, Assam, India")

    if st.button("📍 Find Location"):
        lat, lon = geocode_location(location_query)
        if lat is not None:
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.success("Location found")
        else:
            st.warning("Search failed. Use manual coordinates below.")

    st.divider()
    st.subheader("📌 Manual Coordinates")
    manual_lat = st.number_input("Latitude", value=26.1445, format="%.6f")
    manual_lon = st.number_input("Longitude", value=91.7362, format="%.6f")

    if st.button("📌 Use Manual Coordinates"):
        st.session_state.lat = manual_lat
        st.session_state.lon = manual_lon
        st.success("Coordinates set")

    st.divider()
    predict_clicked = st.button("🚨 Predict Flood Risk")

# -------------------------------------------------
# PREDICTION LOGIC
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
        ["🗺️ Map", "📜 Rule Explanation", "🚨 Alerts", "✨ AI Explanation"]
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
        for e in rule_explanation(**st.session_state.inputs):
            st.markdown(e)

    with tab3:
        for a in safety_recommendations(st.session_state.sim_prediction):
            st.markdown(a)

    with tab4:
        st.markdown(
            PREGENERATED_AI_EXPLANATIONS[
                st.session_state.sim_prediction
            ]
        )
        st.caption(
            "AI explanations were generated using Gemini during development "
            "and cached for reliable demo deployment."
        )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption("⚠️ Academic & demonstration use only • Cloud-safe deployment")
