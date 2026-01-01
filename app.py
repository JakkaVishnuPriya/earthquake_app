import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import uuid

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Earthquake Alert Prediction",
    layout="centered"
)

st.title("🌍 Earthquake Alert Prediction System")
st.write("Predict earthquake alert level using Machine Learning")

# ----------------------------------
# Load Model, Scaler, Feature Names
# ----------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("earthquake_rf_model.pkl")
    scaler = joblib.load("earthquake_scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ----------------------------------
# Alert mappings
# ----------------------------------
alert_to_class = {
    "green": 0,
    "yellow": 1,
    "orange": 2,
    "red": 3
}

class_to_alert = {v: k for k, v in alert_to_class.items()}

ALERT_COLORS = {
    "green": "#10b981",
    "yellow": "#f59e0b",
    "orange": "#f97316",
    "red": "#ef4444"
}

# ----------------------------------
# Input Form
# ----------------------------------
st.subheader("📊 Enter Earthquake Parameters")

with st.form("prediction_form"):
    magnitude = st.number_input("Magnitude", step=0.1)
    depth = st.number_input("Depth (km)", step=1.0)
    cdi = st.number_input("CDI", step=0.1)
    mmi = st.number_input("MMI", step=0.1)
    sig = st.number_input("Significance (sig)", step=1.0)

    submitted = st.form_submit_button("🔮 Predict Alert Level")

# ----------------------------------
# Prediction
# ----------------------------------
if submitted:
    try:
        prediction_id = str(uuid.uuid4())[:8]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        input_data = pd.DataFrame([{
            "magnitude": magnitude,
            "depth": depth,
            "cdi": cdi,
            "mmi": mmi,
            "sig": sig
        }])

        input_data = input_data[feature_names]
        scaled_input = scaler.transform(input_data)

        predicted_class = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]

        predicted_alert = class_to_alert[predicted_class]
        alert_color = ALERT_COLORS[predicted_alert]

        # ----------------------------------
        # Display Result
        # ----------------------------------
        st.markdown(
            f"""
            <div style="padding:15px; background-color:{alert_color}; color:white; border-radius:10px; text-align:center;">
                <h2>ALERT LEVEL: {predicted_alert.upper()}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(f"🆔 **Prediction ID:** {prediction_id}")
        st.write(f"🕒 **Time:** {current_time}")

        # ----------------------------------
        # Probability Chart
        # ----------------------------------
        st.subheader("📈 Prediction Probabilities")

        alerts = [class_to_alert[i].capitalize() for i in range(len(probabilities))]
        plot_colors = ['#10b981', '#f59e0b', '#f97316', '#ef4444']

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(alerts, probabilities, color=plot_colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")

        for i, p in enumerate(probabilities):
            ax.text(i, p + 0.02, f"{p*100:.1f}%", ha='center')

        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
