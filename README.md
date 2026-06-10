# 🌊 FloodWatch - AI-Powered Flood Prediction System

A production-grade Machine Learning web application built with **Streamlit** that predicts flood severity across **4 levels** using 12 real-world environmental indicators - powered by Gradient Boosting with interactive maps, seasonal analysis, and emergency recommendations.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 🌟 Features

| Feature                   | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| 🔍 AI Flood Predictor     | 12-feature Gradient Boosting classifier — 4 severity levels |
| 🗺️ Interactive Risk Map | Geospatial Plotly map showing regional flood distribution    |
| 🎛️ What-If Simulator    | Real-time risk updates as you adjust parameters              |
| 📅 Seasonal Analysis      | Monthly patterns + multi-city comparison + heatmap           |
| 📊 Model Insights         | Feature importance, distributions, correlation matrix        |
| 🚨 Emergency Alerts       | Auto risk alerts + tiered evacuation recommendations         |
| 📡 Radar Profile          | Compare your area vs low-risk and extreme-risk baselines     |
| 🌍 10 Global Cities       | Mumbai, Bangkok, Jakarta, New Orleans, Dhaka & more          |

---

## 🌊 Severity Levels

| Level    | Emoji | Risk Range | Action               |
| -------- | ----- | ---------- | -------------------- |
| Low      | 🟢    | 0–30%     | Monitor only         |
| Moderate | 🟡    | 30–55%    | Prepare precautions  |
| High     | 🟠    | 55–78%    | Consider evacuation  |
| Extreme  | 🔴    | 78–100%   | Evacuate immediately |

---

## 🧬 Input Features (12 Environmental Parameters)

| Feature               | Unit  | Description                           |
| --------------------- | ----- | ------------------------------------- |
| Rainfall              | mm    | 24-hour total precipitation           |
| River Level           | m     | Current river/water body height       |
| Soil Moisture         | %     | Pre-event soil saturation             |
| Elevation             | m     | Terrain height above sea level        |
| Drainage Capacity     | %     | Infrastructure drainage functionality |
| Days Since Last Flood | days  | Time elapsed since last flood event   |
| Temperature           | °C   | Ambient air temperature               |
| Humidity              | %     | Relative atmospheric humidity         |
| Wind Speed            | km/h  | Current wind velocity                 |
| Population Density    | /km² | People per square kilometer           |
| Urban Cover           | %     | Impervious surface percentage         |
| Terrain Slope         | °    | Average land slope gradient           |

---

## 🤖 Model Details

- **Algorithm:** Gradient Boosting Classifier (primary)
- **Secondary:** Random Forest (feature importance)
- **Hyperparameters:** n_estimators=300, max_depth=5, lr=0.08, subsample=0.85
- **Preprocessing:** StandardScaler
- **Classes:** 4 (Low / Moderate / High / Extreme)
- **Train / Test:** 80% / 20% stratified split

---

## 🗂️ Project Structure

```
floodwatch/
├── app.py              ← Full Streamlit application (single file)
├── requirements.txt    ← Python dependencies
└── README.md           ← Project documentation
```

---

## 📸 App Pages

1. **🏠 Home** — Overview, severity system, how-it-works, features
2. **🔍 Flood Predictor** — Full input form → risk level → alerts → radar → recommendations
3. **🗺️ Risk Map** — Regional + global flood hotspot geospatial maps
4. **🎛️ What-If Simulator** — Live parameter sliders → instant risk update
5. **📅 Seasonal Analysis** — Monthly patterns, multi-city comparison, heatmap
6. **📊 Model Insights** — Feature importance, distributions, correlation matrix
7. **ℹ️ About** — Model config, feature dictionary, tech stack

---

## 🌍 Pre-loaded Global Cities

Mumbai · Bangkok · Jakarta · New Orleans · Dhaka ·
Houston · Amsterdam · Colombo · Venice · Lagos

---

## ⚠️ Disclaimer

> FloodWatch is built for **educational and disaster awareness purposes only**.
> Always follow official emergency management and government guidance during flood events.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML:** Scikit-Learn (Gradient Boosting + Random Forest)
- **Visualization:** Plotly / Plotly Express
- **Geospatial:** Plotly Scattergeo
- **Data:** Pandas / NumPy

---

Built with 🌊 | FloodWatch v1.0
