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
### ▶️ Run Locally

# Clone the repository
git clone https://github.com/alroy-24/flood-risk-prediction-dashboard.git
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



```

##🔹 Data Processing & ML Pipeline
```text


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

```
🔹 User Interaction Flow

```text

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

```

## 🤖 Machine Learning Module

The Machine Learning (ML) module is the core intelligence of the system, responsible for learning flood-risk patterns from satellite-derived environmental data and generating predictions used by the dashboard.

---

### 🎯 Objective

The primary objective of the ML module is to:

- Learn relationships between environmental conditions and flood occurrence
- Classify flood risk into:
  - **Low**
  - **Medium**
  - **High**
- Provide a **reliable and interpretable model** suitable for disaster-related decision support

---

### 🛰️ Data Source & Features

#### Data Source
- Satellite-derived data processed using **Google Earth Engine**
- Aggregated environmental features at the location level

#### Input Features
The model uses the following physically meaningful features:

| Feature | Description |
|------|-------------|
| Rainfall | Recent cumulative rainfall |
| Elevation | Height above sea level (DEM) |
| Terrain Slope | Gradient of terrain |
| River Proximity | Binary indicator of nearby river |

These features are chosen due to their **direct influence on flood formation**.

<img width="943" height="583" alt="image" src="https://github.com/user-attachments/assets/61a02430-89ce-45a7-92c6-f25035dc0215" />


---

### 🔄 Data Preprocessing

The preprocessing stage includes:

- Handling missing or inconsistent values
- Selecting relevant environmental features
- Encoding target labels (Low / Medium / High risk)
- Splitting data into training and testing sets

This ensures the model generalizes well to unseen data.

---

### 🧠 Model Selection & Training

- A **supervised classification model** is trained on the processed dataset
- The model is optimized for:
  - Prediction accuracy
  - Robustness across risk classes
  - Interpretability
- Training includes hyperparameter tuning to avoid overfitting

The selected model is well-suited for **tabular environmental data with non-linear relationships**.

---

### 📊 Model Evaluation

Model performance is evaluated using:

- Classification accuracy
- Confusion matrix
- Risk-wise prediction consistency

This validation ensures the model behaves reliably before deployment.

---

### 🔍 Explainability & Interpretability

To avoid black-box behavior:

- Feature contributions are analyzed
- Rule-based explanations are derived from model behavior
- Key factors such as rainfall intensity, elevation, slope, and river proximity are highlighted

This supports **Explainable AI (XAI)**, which is critical for safety-critical systems like flood prediction.

---

### 📦 Model Export & Deployment

- The trained model is serialized using `joblib`
- Saved as a `.pkl` file
- Loaded directly by the Streamlit application for real-time inference

No retraining occurs during deployment, ensuring **fast and stable predictions**.

---

### 🌧️ Scenario-Based Prediction (What-If Simulation)

The ML model supports **what-if rainfall simulations**:

- Rainfall values can be adjusted dynamically
- The same trained model predicts flood risk under hypothetical conditions
- Enables **worst-case scenario analysis and preparedness planning**

---













