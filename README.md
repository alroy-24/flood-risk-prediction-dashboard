# 🌊 Flood Risk Prediction & Decision Support System

An **AI-powered flood risk assessment dashboard** that predicts flood vulnerability, simulates extreme rainfall scenarios, explains risk factors, and provides safety recommendations — built for **reliability, explainability, and real-world deployment**.

---

## 🚀 Project Overview

Flooding is one of the most damaging climate-related disasters, especially in regions like India. While satellite data exists, most flood-warning systems lack:

- Localized predictions  
- Scenario-based analysis  
- Explainable AI outputs  
- Reliable deployment  

This project addresses these gaps by combining **satellite-derived data**, **machine learning**, **what-if simulations**, and **geospatial visualization** into a single decision-support system.

---

## ✨ Key Features

### 📍 Location Handling
- Place-name search (cloud-safe)
- Manual latitude & longitude input (always reliable)
- Designed for satellite-based workflows

---

### 🤖 Flood Risk Prediction (Core AI)
- Predicts **Low / Medium / High flood risk**
- Uses environmental features:
  - Rainfall
  - Elevation
  - Terrain slope
  - River proximity
- Powered by a trained **machine learning classifier**

---

### 🌧️ What-If Rainfall Simulation ⭐
- Simulate increased or decreased rainfall
- Instantly observe changes in flood risk
- Enables **worst-case scenario analysis**
- Transforms prediction into **planning & preparedness**

---

### 🗺️ Interactive Map Visualization
- Displays prediction results spatially
- Helps identify vulnerable regions quickly

---

### 🧠 Explainable AI (Transparency)
- Rule-based explanations for each prediction
- Clearly shows **why** a region is at risk
- Essential for safety-critical systems like disaster management

---

### ✨ AI Explanation Layer (Pre-Generated)
- Natural-language explanations generated during development
- Dynamically displayed based on prediction
- **No live API calls** → stable & demo-safe

---

### 🚨 Alerts & Safety Recommendations
- Contextual safety advice based on risk level
- Bridges the gap between AI output and real-world action

---

## 🧰 Technologies Used (Detailed)

### 🛰️ Google Technologies
- **Google Earth Engine (GEE)**  
  Used for processing satellite-derived rainfall and terrain data such as elevation and slope.  
  Enabled large-scale geospatial analysis for flood-risk modeling.

- **Google Cloud Infrastructure**  
  Deployment backend via Streamlit Cloud.

---

### 🤖 Artificial Intelligence & Machine Learning

- **Machine Learning Model**
  - Supervised classification model
  - Outputs: Low / Medium / High flood risk
  - Input features: rainfall, elevation, slope, river proximity

- **Explainable AI (XAI)**
  - Rule-based logic to interpret model predictions
  - Ensures transparency and trust

- **What-If Scenario Simulation**
  - Evaluates flood risk under hypothetical rainfall changes
  - Enables proactive disaster planning

---

### 🗺️ Visualization & Deployment
- Streamlit
- Folium (interactive maps)
- Python

---

# Clone the repository
git clone https://github.com/YOUR_USERNAME/flood-risk-prediction-dashboard.git
cd flood-risk-prediction-dashboard

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

---

## 🏗️ System Architecture

### 🔹 High-Level Architecture

```text
┌──────────────────────────┐
│   Google Earth Engine    │
│  (Satellite & Terrain)  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Feature Engineering     │
│  • Rainfall              │
│  • Elevation (DEM)       │
│  • Terrain Slope         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Machine Learning Model  │
│  • Flood Risk Classifier │
│  • Low / Medium / High   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Streamlit Dashboard     │
│  • Prediction            │
│  • Simulation            │
│  • Map Visualization     │
│  • Explanations & Alerts │
└──────────────────────────┘
Satellite Data
 (Rainfall, DEM)
        │
        ▼
Google Earth Engine
        │
        ▼
Preprocessing & Aggregation
        │
        ▼
Feature Dataset
        │
        ▼
ML Model Training
        │
        ▼
Saved Model (.pkl)
User
 │
 ▼
Streamlit Web Interface
 │
 ├─ Location Input
 │    • Place Name
 │    • Manual Coordinates
 │
 ├─ Environmental Inputs
 │    • Rainfall
 │    • Elevation
 │    • Slope
 │    • River Proximity
 │
 ├─ What-If Rainfall Simulation
 │
 ▼
Flood Risk Prediction
 │
 ├─ Map Visualization
 ├─ Explainable AI Rules
 ├─ AI Explanation (Pre-generated)
 └─ Safety Recommendations





