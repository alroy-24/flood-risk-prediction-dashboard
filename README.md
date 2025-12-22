# 🌊 Flood Risk Prediction & Decision Support System

An end-to-end AI-powered flood risk prediction system that combines satellite-derived data, machine learning, explainable AI, and interactive geospatial visualization to support disaster preparedness and decision-making.

Built as part of a Google Developer Group (GDG) Hackathon.

---

## 🚀 Key Features

- Flood risk classification using a trained XGBoost model
- Location-based analysis with interactive map visualization
- Explainable AI panel explaining prediction drivers
- What-if rainfall simulation for scenario-based risk assessment
- Risk-based alerts and safety recommendations
- Persistent and interactive Streamlit dashboard

---

## 🛰️ Data & Technology Stack

**Data Sources**
- Satellite-derived rainfall and terrain features
- Terrain elevation, slope, and river proximity (Google Earth Engine)

**Technologies Used**
- Programming: Python
- Machine Learning: XGBoost
- Data Processing: NumPy, Pandas
- Visualization: Streamlit, Folium
- Geocoding: OpenStreetMap (Nominatim API)
- Deployment: Streamlit

---

## 🧠 How the System Works

1. User searches for a geographic location
2. Inputs rainfall and terrain parameters
3. Model predicts flood risk level (Low / Medium / High)
4. System explains key contributing factors
5. Users simulate rainfall changes to assess risk escalation
6. Safety alerts and recommendations are generated

---

## 🌧️ What-If Rainfall Simulation

The system allows users to simulate future rainfall scenarios by adjusting rainfall levels and observing how flood risk changes. This supports:
- Climate uncertainty analysis
- Emergency preparedness
- Risk escalation assessment

---

## 📍 Use Cases

- Disaster risk assessment
- Emergency response planning
- Urban and infrastructure planning
- Climate impact analysis
- Smart city decision-support systems

---


##⚠️ Limitations & Future Improvements

- Currently uses simulated rainfall inputs rather than live forecasts
- Can be extended with real-time weather APIs
- Integration with government flood warning systems
- Expansion to multi-location batch risk analysis

---


▶️ Run Locally
# Clone the repository
git clone https://github.com/alroyp_2405/flood-risk-prediction-dashboard.git
cd flood-risk-prediction-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

##⚠️ Notes

- Ensure Python 3.9 or above is installed

- Internet connection required for location search (OpenStreetMap)

- Model file (flood_xgb_model.pkl) must be present in the project root










