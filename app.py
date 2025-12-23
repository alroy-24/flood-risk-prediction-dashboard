import streamlit as st
import numpy as np
import joblib
import requests
import folium
from streamlit_folium import st_folium
import functools
import time

from google import genai  # Gemini SDK

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
# CLIENTS (OPENCAGE + GEMINI)
# -------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_gemini_client():
    api_key = st.secrets["gemini"]["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key)

# -------------------------------------------------
# GEOCODING WITH OPENCAGE
# -------------------------------------------------
@functools.lru_cache(maxsize=256)
def geocode_location(place: str):
    place = place.strip()
    if not place:
        return None, None

    api_key = st.secrets["geocoding"]["OPENCAGE_KEY"]

    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {
        "q": place,
        "key": api_key,
        "limit": 1,
        "no_annotations": 1,
    }

    for attempt in range(2):
        try:
            r = requests.get(url, params=params, timeout=8)

            if r.status_code == 429:
                time.sleep(2)
                continue

            r.raise_for_status()
            data = r.json()
            if data.get("results"):
                geometry = data["results"][0]["geometry"]
                return float(geometry["lat"]), float(geometry["lng"])
            break

        except requests.exceptions.Timeout:
            time.sleep(1)
        except Exception:
            break

    return None, None

# -------------------------------------------------
# EXPLAINABLE AI (RULE-BASED)
# -------------------------------------------------
def explain_prediction(rainfall, elevation, slope, river):
    exp = []

    if rainfall > 120:
        exp.append("• Very high rainfall significantly increases flood risk.")
    elif rainfall > 80:
        exp.append("• Moderate rainfall contributes to surface water accumulation.")

    if elevation < 30:
        exp.append("• Low elevation makes the area flood-prone.")
    elif elevation < 50:
        exp.append("• Relatively low elevation increases vulnerability.")

    if river == 1:
        exp.append("• Proximity to a river increases overflow probability.")

    if slope < 1:
        exp.append("• Flat terrain slows natural drainage.")

    return exp

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
            "• Monitor rainfall and river-level alerts."
        ]
    else:
        return [
            "• High flood risk detected.",
            "• Prepare for evacuation if advised.",
            "• Avoid travel near rivers and waterlogged areas.",
            "• Follow official disaster-management advisories."
        ]

# -------------------------------------------------
# GEMINI-BASED EXPLANATION
# -------------------------------------------------
def gemini_explanation(pred_level, features):
    ...
    client = get_gemini_client()
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return resp.text.strip()


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.sim_prediction = None
    st.session_state.lat = None
    st.session_state.lon = None
    st.session_state.inputs = {}
    st.session_state.location_label = "Selected location"

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
# 🌊 Flood Risk Prediction Dashboard
**AI-powered flood risk assessment using satellite-derived data and machine learning**
""")
st.caption("Built using Google Earth Engine, XGBoost, Streamlit, OpenCage & Gemini")
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
    rainfall_delta = st.slider("Rainfall change (mm)", -50, 100, 0)
    simulated_rainfall = max(0, rainfall + rainfall_delta)
    st.caption(f"Scenario rainfall: **{simulated_rainfall} mm**")

    st.divider()

    # -------- LOCATION SECTION --------
    st.subheader("📍 Location Search (Best Effort)")
    location_query = st.text_input(
        "City, State, Country",
        "Guwahati, Assam, India",
        help="Example: 'Mumbai, Maharashtra, India'"
    )

    if st.button("📍 Find / Refresh Location"):
        with st.spinner("Looking up location..."):
            lat, lon = geocode_location(location_query)

        if lat is not None and lon is not None:
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.session_state.location_label = location_query
            st.success(f"Location found: {lat:.4f}, {lon:.4f}")
        else:
            st.info(
                "Could not automatically find this place. "
                "You can still enter or adjust the coordinates below."
            )

    st.divider()

    st.subheader("📌 Coordinates (Auto-filled)")
    manual_lat = st.number_input(
        "Latitude",
        value=st.session_state.lat or 26.1445,
        format="%.6f"
    )
    manual_lon = st.number_input(
        "Longitude",
        value=st.session_state.lon or 91.7362,
        format="%.6f"
    )

    if st.button("✅ Use These Coordinates"):
        st.session_state.lat = manual_lat
        st.session_state.lon = manual_lon
        st.success("Coordinates set successfully")

    st.divider()

    predict_clicked = st.button("🚨 Predict Flood Risk")

# -------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------
if predict_clicked:
    if st.session_state.lat is None or st.session_state.lon is None:
        st.error(
            "Please set a location first using 'Find / Refresh Location' "
            "or by confirming the coordinates."
        )
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

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Current Conditions")
        if st.session_state.prediction == 0:
            st.success("🟢 LOW RISK")
        elif st.session_state.prediction == 1:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

    with c2:
        st.markdown("### Simulated Scenario")
        if st.session_state.sim_prediction == 0:
            st.success("🟢 LOW RISK")
        elif st.session_state.sim_prediction == 1:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

    # Extra Gemini tab
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Map", "🧠 Rules Explanation", "🚨 Alerts", "🤖 Gemini Explanation"]
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
            fill=True,
            fill_opacity=0.85
        ).add_to(m)

        st_folium(m, width=900, height=450)

    with tab2:
        for e in explain_prediction(
            st.session_state.inputs["rainfall"],
            st.session_state.inputs["elevation"],
            st.session_state.inputs["slope"],
            st.session_state.inputs["river"]
        ):
            st.markdown(e)

    with tab3:
        for a in safety_recommendations(st.session_state.sim_prediction):
            st.markdown(a)

    with tab4:
        with st.spinner("Generating AI explanation with Gemini..."):
            g_text = gemini_explanation(
                st.session_state.sim_prediction,
                {
                    "rainfall": st.session_state.inputs["rainfall"],
                    "elevation": st.session_state.inputs["elevation"],
                    "slope": st.session_state.inputs["slope"],
                    "river": st.session_state.inputs["river"],
                    "location": st.session_state.location_label,
                },
            )
        st.markdown(g_text)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption(
    "⚠️ Academic & demonstration use only • "
    "Designed for cloud-safe deployment"
)
