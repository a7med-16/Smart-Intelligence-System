import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import sqlite3

from datetime import datetime
from scipy.sparse import hstack, csr_matrix

import plotly.graph_objects as go

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Incident SaaS Platform",
    page_icon="🚨",
    layout="wide"
)

# =========================
# CUSTOM STYLE
# =========================
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

h1, h2, h3 {
    color: white;
}

.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
}

.stButton > button {

    width: 100%;
    border-radius: 10px;
    height: 3em;

    background: linear-gradient(
        90deg,
        #ff4b4b,
        #ff884b
    );

    color: white;
    border: none;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================
# DATABASE
# =========================
conn = sqlite3.connect(
    "saas.db",
    check_same_thread=False
)

c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (

    username TEXT PRIMARY KEY,
    password TEXT

)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS predictions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    timestamp TEXT,
    severity INTEGER,
    confidence REAL,
    risk REAL

)
""")

conn.commit()

# =========================
# AUTH FUNCTIONS
# =========================
def register_user(username, password):

    try:

        c.execute(
            "INSERT INTO users VALUES (?,?)",
            (username, password)
        )

        conn.commit()

        return True

    except:

        return False


def login_user(username, password):

    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    return c.fetchone()

# =========================
# LOAD MODEL ASSETS
# =========================
@st.cache_resource
def load_assets():

    model = joblib.load(
        "xgb_accident_model.pkl"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    features = joblib.load(
        "features.pkl"
    )

    encoders = joblib.load(
        "label_encoders.pkl"
    )

    return (
        model,
        scaler,
        vectorizer,
        features,
        encoders
    )

model, scaler, vectorizer, features, encoders = load_assets()

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text

# =========================
# SESSION STATE
# =========================
if "user" not in st.session_state:

    st.session_state.user = None
# =========================
# SIDEBAR
# =========================
st.sidebar.success(
    f"Logged in as: {st.session_state.user}"
)

if st.sidebar.button("Logout"):

    st.session_state.user = None

    st.rerun()

st.sidebar.header(
    "📍 Incident Inputs"
)

# =========================
# LOCATION
# =========================
start_lat = st.sidebar.number_input(
    "Latitude",
    value=40.0
)

start_lng = st.sidebar.number_input(
    "Longitude",
    value=-80.0
)

# =========================
# WEATHER
# =========================
distance = st.sidebar.number_input(
    "Distance (mi)",
    value=1.0
)

temp = st.sidebar.number_input(
    "Temperature (F)",
    value=70.0
)

humidity = st.sidebar.number_input(
    "Humidity (%)",
    value=50.0
)

pressure = st.sidebar.number_input(
    "Pressure (in)",
    value=30.0
)

visibility = st.sidebar.number_input(
    "Visibility (mi)",
    value=10.0
)

wind_speed = st.sidebar.number_input(
    "Wind Speed (mph)",
    value=10.0
)

precip = st.sidebar.number_input(
    "Precipitation (in)",
    value=0.0
)

# =========================
# TIME
# =========================
hour = st.sidebar.number_input(
    "Hour",
    value=12
)

day = st.sidebar.number_input(
    "Day",
    value=1
)

month = st.sidebar.number_input(
    "Month",
    value=6
)

# =========================
# ROAD CONDITIONS
# =========================
traffic_signal = st.sidebar.selectbox(
    "Traffic Signal",
    [0, 1]
)

stop = st.sidebar.selectbox(
    "Stop",
    [0, 1]
)

junction = st.sidebar.selectbox(
    "Junction",
    [0, 1]
)

crossing = st.sidebar.selectbox(
    "Crossing",
    [0, 1]
)

railway = st.sidebar.selectbox(
    "Railway",
    [0, 1]
)

roundabout = st.sidebar.selectbox(
    "Roundabout",
    [0, 1]
)

# =========================
# TEXT FEATURES
# =========================
weather = st.sidebar.text_input(
    "Weather Condition",
    "Clear"
)

wind_dir = st.sidebar.text_input(
    "Wind Direction",
    "N"
)

sunset = st.sidebar.text_input(
    "Sunrise/Sunset",
    "Day"
)

description = st.text_area(
    "📝 Incident Description",
    "Accident on highway"
)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_all():

    input_dict = {

        "Start_Lat": start_lat,
        "Start_Lng": start_lng,

        "Distance(mi)": distance,

        "Temperature(F)": temp,
        "Humidity(%)": humidity,
        "Pressure(in)": pressure,
        "Visibility(mi)": visibility,
        "Wind_Speed(mph)": wind_speed,
        "Precipitation(in)": precip,

        "Hour": hour,
        "Day": day,
        "Month": month,

        "Traffic_Signal": traffic_signal,
        "Stop": stop,
        "Junction": junction,
        "Crossing": crossing,
        "Railway": railway,
        "Roundabout": roundabout,

        "Weather_Condition": weather,
        "Wind_Direction": wind_dir,
        "Sunrise_Sunset": sunset
    }

    # =====================
    # DATAFRAME
    # =====================
    df = pd.DataFrame([input_dict])

    # =====================
    # ENCODING
    # =====================
    for col, le in encoders.items():

        if col in df.columns:

            try:

                df[col] = le.transform(
                    df[col].astype(str)
                )

            except:

                df[col] = 0

    # =====================
    # FEATURE ALIGNMENT
    # =====================
    df = df.reindex(
        columns=features,
        fill_value=0
    )

    # =====================
    # SCALING
    # =====================
    X_num = scaler.transform(df)

    # =====================
    # TEXT PROCESSING
    # =====================
    clean_desc = clean_text(
        description
    )

    X_text = vectorizer.transform(
        [clean_desc]
    )

    # =====================
    # MERGE FEATURES
    # =====================
    X = hstack([

        csr_matrix(X_num),

        X_text

    ])

    # =====================
    # PREDICTION
    # =====================
    pred = model.predict(X)[0]

    proba = model.predict_proba(X)[0]

    risk = float(max(proba))

    # =====================
    # SAVE TO DATABASE
    # =====================
    c.execute("""

    INSERT INTO predictions
    VALUES (NULL,?,?,?,?,?)

    """, (

        st.session_state.user,

        str(datetime.now()),

        int(pred + 1),

        float(max(proba)),

        risk

    ))

    conn.commit()

    return pred, proba, risk

# =========================
# MAIN UI
# =========================
st.title(
    "🚨 AI Incident SaaS Dashboard"
)

st.caption(
    "AI-powered incident severity prediction platform"
)

# =========================
# RUN PREDICTION
# =========================
if st.button("🚀 Run Prediction"):

    with st.spinner(
        "Analyzing incident using AI..."
    ):

        pred, proba, risk = predict_all()

    # =====================
    # METRICS
    # =====================
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Severity",
        f"S{int(pred + 1)}"
    )

    col2.metric(
        "Confidence",
        f"{max(proba)*100:.2f}%"
    )

    col3.metric(
        "Risk Score",
        f"{risk:.3f}"
    )

    # =====================
    # RISK MESSAGE
    # =====================
    if risk > 0.7:

        st.error(
            "🔴 HIGH RISK INCIDENT DETECTED"
        )

    elif risk > 0.4:

        st.warning(
            "🟠 MEDIUM RISK INCIDENT"
        )

    else:

        st.success(
            "🟢 LOW RISK INCIDENT"
        )

    # =====================
    # GAUGE CHART
    # =====================
    fig = go.Figure(go.Indicator(

        mode="gauge+number",

        value=risk * 100,

        title={
            'text': "Risk Level"
        },

        gauge={

            'axis': {
                'range': [0, 100]
            },

            'bar': {
                'color': "red"
            },

            'steps': [

                {
                    'range': [0, 40],
                    'color': "green"
                },

                {
                    'range': [40, 70],
                    'color': "orange"
                },

                {
                    'range': [70, 100],
                    'color': "red"
                }
            ]
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================
    # CLASS PROBABILITIES
    # =====================
    st.subheader(
        "📊 Class Probabilities"
    )

    probs_df = pd.DataFrame({

        "Class": [
            "S1",
            "S2",
            "S3",
            "S4"
        ],

        "Probability": [

            proba[0],
            proba[1],
            proba[2],
            proba[3]

        ]
    })

    st.bar_chart(
        probs_df.set_index("Class")
    )

# =========================
# USER HISTORY
# =========================
st.markdown(
    "## 📜 Your Prediction History"
)

history = pd.read_sql_query(

    f"""
    SELECT *
    FROM predictions
    WHERE username='{st.session_state.user}'
    ORDER BY id DESC
    """,

    conn
)

st.dataframe(
    history,
    use_container_width=True
)

# =========================
# ANALYTICS
# =========================
st.markdown(
    "## 📈 System Analytics"
)

all_data = pd.read_sql_query(
    "SELECT * FROM predictions",
    conn
)

if len(all_data) > 0:

    analytics_col1, analytics_col2 = st.columns(2)

    with analytics_col1:

        st.metric(
            "Total Predictions",
            len(all_data)
        )

    with analytics_col2:

        st.metric(
            "Average Confidence",
            round(
                all_data["confidence"].mean(),
                3
            )
        )

    st.subheader(
        "Severity Distribution"
    )

    st.bar_chart(
        all_data["severity"].value_counts()
    )
    