import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
<style>

.main{
    background-color:#f5f7fb;
    margin: 5px;
}

.block-container{
    padding-top:2.5rem;
    padding-bottom:2rem;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#d32f2f;
}

.subtitle{
    text-align:center;
    color:#666;
    font-size:18px;
    margin-bottom:20px;
}

div[data-testid="stMetric"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}

</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# LOAD MODEL
# =====================================================
# =====================================================
# LOAD MODEL
# =====================================================


@st.cache_resource
def load_model():
    model = joblib.load("Model/model.pkl")
    scaler = joblib.load("Model/scaler.pkl")
    feature_names = joblib.load("feature_names.joblib")
    return model, scaler, feature_names


try:
    model, scaler, feature_names = load_model()
except Exception as e:
    st.error(f"Unable to load model.\n\n{e}")
    st.stop()
# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<p class="title" >❤️ Heart Disease Prediction</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">Predict the likelihood of heart disease using clinical measurements.</p>',
    unsafe_allow_html=True,
)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("🩺 Patient Information")

age = st.sidebar.slider("Age", 20, 90, 50)

sex = st.sidebar.selectbox(
    "Sex",
    [0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male",
)

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3],
    help="0=Typical Angina, 1=Atypical Angina, 2=Non-anginal Pain, 3=Asymptomatic",
)

trestbps = st.sidebar.slider(
    "Resting Blood Pressure",
    80,
    220,
    120,
)

chol = st.sidebar.slider(
    "Serum Cholesterol",
    100,
    600,
    220,
)

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar >120 mg/dl",
    [0, 1],
)

restecg = st.sidebar.selectbox(
    "Resting ECG",
    [0, 1, 2],
)

thalach = st.sidebar.slider(
    "Maximum Heart Rate",
    60,
    220,
    150,
)

exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    [0, 1],
)

oldpeak = st.sidebar.slider(
    "Old Peak",
    0.0,
    7.0,
    1.0,
)

slope = st.sidebar.selectbox(
    "Slope",
    [0, 1, 2],
)

ca = st.sidebar.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4],
)

thal = st.sidebar.selectbox(
    "Thalassemia",
    [0, 1, 2, 3],
)

# =====================================================
# PREDICTION BUTTON
# =====================================================

predict = st.sidebar.button("🔍 Predict Heart Disease", use_container_width=True)

# =====================================================
# PREDICTION
# =====================================================

if predict:

    input_data = np.array(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
        ]]
    )

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    left, right = st.columns([1.2, 1])

    # =====================================================
    # LEFT PANEL
    # =====================================================

    with left:

        st.subheader("📋 Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        st.metric(
            "Risk Probability",
            f"{probability * 100:.2f}%",
        )

        st.progress(float(probability))

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#d32f2f"},
                    "steps": [
                        {"range": [0, 40], "color": "#81c784"},
                        {"range": [40, 70], "color": "#ffb74d"},
                        {"range": [70, 100], "color": "#ef5350"},
                    ],
                },
            )
        )

        gauge.update_layout(height=350)

        st.plotly_chart(gauge, use_container_width=True)

    # =====================================================
    # RIGHT PANEL
    # =====================================================

    with right:

        st.subheader("🧾 Patient Summary")

        summary = pd.DataFrame(
            {
                "Feature": [
                    "Age",
                    "Sex",
                    "Chest Pain",
                    "Blood Pressure",
                    "Cholesterol",
                    "Heart Rate",
                    "Exercise Angina",
                    "Old Peak",
                ],
                "Value": [
                    age,
                    "Male" if sex else "Female",
                    cp,
                    trestbps,
                    chol,
                    thalach,
                    "Yes" if exang else "No",
                    oldpeak,
                ],
            }
        )

        st.dataframe(summary, use_container_width=True)

    st.divider()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader("📊 Most Important Health Indicators")

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_[0],
        }
    )

    importance["Absolute"] = importance["Coefficient"].abs()

    importance = importance.sort_values(
        by="Absolute",
        ascending=False,
    )

    fig = px.bar(
        importance,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale="RdBu",
        title="Logistic Regression Feature Importance",
    )

    fig.update_layout(height=550)

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =====================================================
    # TOP HEALTH INDICATORS
    # =====================================================

    st.subheader("💡 Top Health Indicators")

    top5 = importance.head(5)

    for _, row in top5.iterrows():
        direction = (
            "Increases"
            if row["Coefficient"] > 0
            else "Decreases"
        )

        st.write(
            f"**{row['Feature']}** → {direction} the likelihood of heart disease."
        )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.info(
    """
### About this Application

This application uses a **Logistic Regression** model trained on the Heart Disease dataset to estimate the likelihood of heart disease based on patient clinical measurements.

**Note:** This tool is intended for educational purposes only and should not be used as a substitute for professional medical diagnosis or advice.
"""
)