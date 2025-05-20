# Re-saving the updated Streamlit app code due to kernel reset

updated_code = '''
import streamlit as st
import random, time, pandas as pd

st.set_page_config(
    page_title="Triksha Health Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ——— Style for blue title ———
st.markdown("""
    <style>
    .blue-title {
        color: #1f77b4;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .stMarkdown h3 {
        margin-top: 0.5em;
    }
    </style>
""", unsafe_allow_html=True)

# ——— Helper for colored status ———
def status_line(msg, level):
    colors = {
        "ok": "#1f77b4",       # blue
        "warning": "#ff7f0e",  # orange
        "critical": "#d62728", # red
    }
    color = colors[level]
    st.markdown(f"<span style='color:{color}; font-weight:bold'>{msg}</span>", unsafe_allow_html=True)

# ——— Simulation utility ———
def simulate_vital(name, unit, lo, hi, threshold=None):
    data = []
    chart = st.empty()
    info = st.empty()
    v_input = st.number_input(f"Manual {name} input ({unit.strip()})", min_value=lo, max_value=hi, value=(lo+hi)//2.0, step=1.0)
    if st.button("Start Simulation"):
        for _ in range(50):
            v = round(random.uniform(lo, hi), 2)
            data.append(v)
            chart.line_chart(pd.DataFrame(data, columns=[name]))
            with info.container():
                st.subheader("🔍 Insights")
                if threshold:
                    if v > threshold:
                        lvl, msg = "critical", f"⚠️ {name} high ({v}{unit})"
                    elif v > threshold * 0.8:
                        lvl, msg = "warning", f"🟠 {name} nearing ({v}{unit})"
                    else:
                        lvl, msg = "ok", f"✔️ {name} OK ({v}{unit})"
                else:
                    lvl, msg = "ok", f"✔️ {name}: {v}{unit}"
                status_line(msg, lvl)
                if lvl == "critical":
                    st.write("• Reason: Exceeds safe range\n• Consensus: Multiple high readings\n• Trend: Rising")
                elif lvl == "warning":
                    st.write("• Reason: Near upper bound\n• Trend: Slightly rising")
                else:
                    st.write("• Trend: Stable")
            time.sleep(0.1)

# ——— Pages ———
def home():
    st.markdown('<div class="blue-title">🏥 Health Insights Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Welcome to Triksha")
    st.write("AI-powered real-time vital monitoring.")
    st.write("---")

    age = st.slider("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age_group = st.selectbox("Age Group", ["Child", "Adult", "Senior"])
    athlete_status = st.selectbox("Athlete Status", ["Athlete", "Non-Athlete"])
    category = st.selectbox("Category", ["Diabetic", "Cardiac", "Respiratory", "General"])
    activity = st.selectbox("Activity Level", ["Resting", "Active"])

    st.info("👈 Use the sidebar to explore vitals.")

def heart():        st.title("❤️ Heart Rate");        simulate_vital("Heart Rate", " bpm", 60, 130, 100)
def hrv():          st.title("💓 HRV");               simulate_vital("HR Variability", " ms", 20, 120, 80)
def resp():         st.title("🌬️ Resp Rate");         simulate_vital("Respiratory Rate", " br/min", 12, 25, 20)
def spo2():         st.title("🧪 SpO₂");              simulate_vital("SpO2", " %", 90, 100, 94)
def bp():
    st.title("🩸 Blood Pressure")
    sys, dia = [], []
    ch = st.empty()
    inf = st.empty()
    if st.button("Start BP Simulation"):
        for _ in range(50):
            s, d = round(random.uniform(100, 180), 2), round(random.uniform(60, 100), 2)
            sys.append(s)
            dia.append(d)
            ch.line_chart(pd.DataFrame({"Systolic": sys, "Diastolic": dia}))
            with inf.container():
                st.subheader("🔍 Insights")
                if s > 140 or d > 90:
                    status_line(f"⚠️ BP high: {s}/{d}", "critical")
                elif s > 112 or d > 72:
                    status_line(f"🟠 BP near: {s}/{d}", "warning")
                else:
                    status_line(f"✔️ BP OK: {s}/{d}", "ok")
                st.write("• Trend:", "Rising" if s > 140 or d > 90 else "Stable")
            time.sleep(0.1)

def glucose():      st.title("🧃 Glucose");            simulate_vital("Glucose", " mg/dL", 70, 200, 140)
def temp():         st.title("🌡️ Temperature");       simulate_vital("Body Temperature", " °C", 36, 39.5, 37.5)
def stress():       st.title("😰 Stress");            simulate_vital("Stress Level", "/10", 1, 10, 7)
def sleep():        st.title("🛌 Sleep");             simulate_vital("Sleep Duration", " hrs", 0, 10, 5)

# ——— Page Routing ———
PAGES = {
    "Home": home, "Heart Rate": heart, "HRV": hrv,
    "Resp Rate": resp, "SpO2": spo2, "Blood Pressure": bp,
    "Glucose": glucose, "Temperature": temp,
    "Stress": stress, "Sleep": sleep
}

st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox("Go to", list(PAGES.keys()))
PAGES[page]()

