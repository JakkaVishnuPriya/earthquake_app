import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import uuid

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Quake Pred | Earthquake Alert Prediction",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# SESSION STATE (LOGIN)
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("earthquake_rf_model.pkl")
    scaler = joblib.load("earthquake_scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# =====================================================
# ALERT CONFIG
# =====================================================
alert_to_class = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
class_to_alert = {v: k for k, v in alert_to_class.items()}

ALERT_COLORS = {
    "green": "#10b981",
    "yellow": "#f59e0b",
    "orange": "#f97316",
    "red": "#ef4444"
}

# =====================================================
# GLOBAL CSS (BEAUTIFUL UI)
# =====================================================
st.markdown("""
<style>
.card {
    padding:20px;
    border-radius:15px;
    background:#0f172a;
    color:white;
    box-shadow:0 10px 25px rgba(0,0,0,0.25);
    margin-bottom:20px;
}
.footer {
    text-align:center;
    padding:20px;
    margin-top:40px;
    background:#020617;
    color:white;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("🌍 Quake Pred")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Try Free", "Predict", "Dashboard", "Login / Logout"]
)

# =====================================================
# HOME PAGE
# =====================================================
if menu == "Home":
    st.markdown("""
    <div class="card">
        <h1>🌍 Quake Pred</h1>
        <p>Advanced Earthquake Alert Prediction System using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("Alert Levels", "Green | Yellow | Orange | Red")
    col3.metric("Deployment", "Streamlit Cloud")

    st.info("🔐 Login to access full prediction & dashboard features.")

# =====================================================
# TRY FREE (NO LOGIN REQUIRED)
# =====================================================
elif menu == "Try Free":
    st.subheader("⚡ Free Earthquake Alert Prediction")

    with st.form("free_form"):
        magnitude = st.number_input("Magnitude", step=0.1)
        depth = st.number_input("Depth (km)", step=1.0)
        cdi = st.number_input("CDI", step=0.1)
        mmi = st.number_input("MMI", step=0.1)
        sig = st.number_input("Significance", step=1.0)
        submit = st.form_submit_button("🔮 Predict")

    if submit:
        X = pd.DataFrame([[magnitude, depth, cdi, mmi, sig]], columns=feature_names)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        alert = class_to_alert[pred]

        st.markdown(
            f"<div class='card' style='background:{ALERT_COLORS[alert]};'>"
            f"<h2>ALERT LEVEL: {alert.upper()}</h2></div>",
            unsafe_allow_html=True
        )

# =====================================================
# LOGIN / LOGOUT
# =====================================================
elif menu == "Login / Logout":
    if not st.session_state.logged_in:
        st.subheader("🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "demo_user" and password == "demo123":
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful")
            else:
                st.error("❌ Invalid credentials")

    else:
        st.success(f"Logged in as **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.warning("Logged out successfully")

# =====================================================
# PREDICT (LOGIN REQUIRED)
# =====================================================
elif menu == "Predict":
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login to access prediction.")
    else:
        st.subheader("📊 Earthquake Alert Prediction")

        with st.form("predict_form"):
            magnitude = st.number_input("Magnitude", step=0.1)
            depth = st.number_input("Depth (km)", step=1.0)
            cdi = st.number_input("CDI", step=0.1)
            mmi = st.number_input("MMI", step=0.1)
            sig = st.number_input("Significance", step=1.0)
            submit = st.form_submit_button("Predict")

        if submit:
            prediction_id = str(uuid.uuid4())[:8]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            X = pd.DataFrame([[magnitude, depth, cdi, mmi, sig]], columns=feature_names)
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)[0]
            probs = model.predict_proba(X_scaled)[0]
            alert = class_to_alert[pred]

            st.markdown(
                f"<div class='card' style='background:{ALERT_COLORS[alert]};'>"
                f"<h2>ALERT LEVEL: {alert.upper()}</h2></div>",
                unsafe_allow_html=True
            )

            st.write(f"🆔 Prediction ID: {prediction_id}")
            st.write(f"🕒 Time: {current_time}")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Green", "Yellow", "Orange", "Red"], probs, color=list(ALERT_COLORS.values()))
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")

            for i, p in enumerate(probs):
                ax.text(i, p + 0.02, f"{p*100:.1f}%", ha="center")

            st.pyplot(fig)

# =====================================================
# DASHBOARD
# =====================================================
elif menu == "Dashboard":
    if not st.session_state.logged_in:
        st.warning("⚠️ Login required.")
    else:
        st.subheader("📈 Dashboard")

        col1, col2, col3 = st.columns(3)
        col1.metric("Predictions Today", "24")
        col2.metric("High Risk Alerts", "5")
        col3.metric("System Status", "Online")

        st.success("System functioning normally.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    <h4>Quake Pred</h4>
    <p>ML-powered Earthquake Alert Prediction System</p>
    <p>© 2024 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
