import streamlit as st
import random, time, pandas as pd
import altair as alt
from collections import deque
import torch
import numpy as np
import math
import torch.nn as nn

LSTM_WINDOW_SIZE = 30
BP_HISTORY = deque(maxlen=40)
HR_HISTORY = deque(maxlen=20)
TEMP_HISTORY = deque(maxlen=40)
TEMP_SEQUENCE = deque(maxlen=20)
RR_HISTORY = deque(maxlen=1000)
HRV_HISTORY = deque(maxlen=40)
BP_THRESHOLDS = {
    "adult": {
        "normal": ((0, 119), (0, 79)),
        "elevated": ((120, 129), (0, 79)),
        "stage1": ((130, 139), (80, 89)),
        "crisis": ((181, float('inf')), (121, float('inf'))),
        "stage2": ((140, 180), (90, 120))
    },
    "older_adult": {
        "normal": ((0, 129), (0, 79)),
        "elevated": ((130, 139), (0, 79)),
        "stage1": ((140, 149), (80, 89)),
        "crisis": ((181, float('inf')), (121, float('inf'))),
        "stage2": ((150, 180), (90, 120))
    },
    "athlete_endurance": {
        "normal": ((0, 115), (0, 75)),
        "elevated": ((116, 125), (0, 75)),
        "stage1": ((126, 135), (76, 85)),
        "crisis": ((181, float('inf')), (121, float('inf'))),
        "stage2": ((136, 180), (86, 120))
    },
    "athlete_strength": {
        "normal": ((0, 120), (0, 80)),
        "elevated": ((121, 129), (0, 79)),
        "stage1": ((130, 139), (80, 89)),
        "crisis": ((181, float('inf')), (121, float('inf'))),
        "stage2": ((140, 180), (90, 120))
    }
}

TEMP_THRESHOLDS = {
    "adolescent": {
        "normal": (97.6, 99.6),
        "fever_mild": (99.1, 100.4),
        "fever_moderate": (100.5, 102.2),
        "fever_severe": (102.3, float('inf')),
        "hypothermia_mild": (93.0, 95.0),
        "hypothermia_moderate": (90.0, 92.9),
        "hypothermia_severe": (0, 89.9)
    },
    "adult": {
        "normal": (97.6, 99.6),
        "fever_mild": (99.1, 100.4),
        "fever_moderate": (100.5, 102.2),
        "fever_severe": (102.3, float('inf')),
        "hypothermia_mild": (93.0, 95.0),
        "hypothermia_moderate": (90.0, 92.9),
        "hypothermia_severe": (0, 89.9)
    },
    "older_adult": {
        "normal": (96.4, 98.5),
        "fever_mild": (98.6, 99.9),
        "fever_moderate": (100.0, 102.0),
        "fever_severe": (102.3, float('inf')),
        "hypothermia_mild": (93.0, 95.0),
        "hypothermia_moderate": (90.0, 92.9),
        "hypothermia_severe": (0, 89.9)
    },
    "athlete_resting": {
        "normal": (97.7, 99.5),
        "fever_mild": (99.6, 100.4),
        "fever_moderate": (100.5, 102.2),
        "fever_severe": (102.3, float('inf')),
        "hypothermia_mild": (93.0, 95.0),
        "hypothermia_moderate": (90.0, 92.9),
        "hypothermia_severe": (0, 89.9)
    },
    "athlete_physical": {
        "normal": (98.6, 102.2),
        "elevated": (102.3, 104.0),
        "heat_stress_mild": (104.1, 104.8),
        "heat_stress_moderate": (104.9, 105.7),
        "heat_stress_severe": (105.8, float('inf')),
        "hypothermia_mild": (93.0, 95.0),
        "hypothermia_moderate": (90.0, 92.9),
        "hypothermia_severe": (0, 89.9)
    }
}
HRV_THRESHOLDS = {
    "adolescent": {
        "resting": {
            "sdnn": {
                "normal": (60, 85),
                "mildly_abnormal": (50, 59),
                "moderately_abnormal": (30, 49),
                "severely_abnormal": (0, 29)
            },
            "rmssd": {
                "normal": (55, 100),
                "mildly_abnormal": (40, 54),
                "moderately_abnormal": (25, 39),
                "severely_abnormal": (0, 24)
            },
            "lf_hf": {
                "normal": (0.6, 1.5),
                "mildly_abnormal": (1.6, 2.0),
                "moderately_abnormal": (2.1, 3.0),
                "severely_abnormal": (3.1, float('inf'))
            },
            "lf": {
                "normal": (700, 1800),
                "mildly_abnormal": (500, 699),
                "moderately_abnormal": (300, 499),
                "severely_abnormal": (0, 299)
            },
            "hf": {
                "normal": (800, 2500),
                "mildly_abnormal": (500, 799),
                "moderately_abnormal": (300, 499),
                "severely_abnormal": (0, 299)
            }
        },
        "active": {
            "sdnn": {
                "normal": (80, 170),
                "mildly_abnormal": (60, 79),
                "moderately_abnormal": (40, 59),
                "severely_abnormal": (0, 39)
            },
            "rmssd": {
                "normal": (100, 300),
                "mildly_abnormal": (60, 99),
                "moderately_abnormal": (30, 59),
                "severely_abnormal": (0, 29)
            },
            "lf_hf": {
                "normal": (0.6, 1.5),
                "mildly_abnormal": (1.6, 2.0),
                "moderately_abnormal": (2.1, 3.0),
                "severely_abnormal": (3.1, float('inf'))
            },
            "lf": {
                "normal": (700, 1800),
                "mildly_abnormal": (500, 699),
                "moderately_abnormal": (300, 499),
                "severely_abnormal": (0, 299)
            },
            "hf": {
                "normal": (800, 2500),
                "mildly_abnormal": (500, 799),
                "moderately_abnormal": (300, 499),
                "severely_abnormal": (0, 299)
            }
        }
    },
    "adult": {
        "resting": {
            "sdnn": {
                "normal": (35, 65),
                "mildly_abnormal": (25, 34),
                "moderately_abnormal": (15, 24),
                "severely_abnormal": (0, 14)
            },
            "rmssd": {
                "normal": (25, 45),
                "mildly_abnormal": (15, 24),
                "moderately_abnormal": (10, 14),
                "severely_abnormal": (0, 9)
            },
            "lf_hf": {
                "normal": (1.0, 2.5),
                "mildly_abnormal": (2.6, 3.5),
                "moderately_abnormal": (3.6, 5.0),
                "severely_abnormal": (5.1, float('inf'))
            },
            "lf": {
                "normal": (300, 800),
                "mildly_abnormal": (150, 299),
                "moderately_abnormal": (75, 149),
                "severely_abnormal": (0, 74)
            },
            "hf": {
                "normal": (200, 600),
                "mildly_abnormal": (100, 199),
                "moderately_abnormal": (50, 99),
                "severely_abnormal": (0, 49)
            }
        },
        "active": {
            "sdnn": {
                "normal": (100, 180),
                "mildly_abnormal": (80, 99),
                "moderately_abnormal": (60, 79),
                "severely_abnormal": (0, 59)
            },
            "rmssd": {
                "normal": (150, 350),
                "mildly_abnormal": (100, 149),
                "moderately_abnormal": (60, 99),
                "severely_abnormal": (0, 59)
            },
            "lf_hf": {
                "normal": (1.0, 2.5),
                "mildly_abnormal": (2.6, 3.5),
                "moderately_abnormal": (3.6, 5.0),
                "severely_abnormal": (5.1, float('inf'))
            },
            "lf": {
                "normal": (300, 800),
                "mildly_abnormal": (150, 299),
                "moderately_abnormal": (75, 149),
                "severely_abnormal": (0, 74)
            },
            "hf": {
                "normal": (200, 600),
                "mildly_abnormal": (100, 199),
                "moderately_abnormal": (50, 99),
                "severely_abnormal": (0, 49)
            }
        }
    },
    "athlete": {
        "resting": {
            "sdnn": {
                "normal": (80, 140),
                "mildly_abnormal": (65, 79),
                "moderately_abnormal": (45, 64),
                "severely_abnormal": (0, 44)
            },
            "rmssd": {
                "normal": (60, 120),
                "mildly_abnormal": (40, 59),
                "moderately_abnormal": (25, 39),
                "severely_abnormal": (0, 24)
            },
            "lf_hf": {
                "normal": (0.5, 1.2),
                "mildly_abnormal": (1.3, 1.8),
                "moderately_abnormal": (1.9, 2.5),
                "severely_abnormal": (2.6, float('inf'))
            },
            "lf": {
                "normal": (500, 1400),
                "mildly_abnormal": (350, 499),
                "moderately_abnormal": (200, 349),
                "severely_abnormal": (0, 199)
            },
            "hf": {
                "normal": (600, 1600),
                "mildly_abnormal": (400, 599),
                "moderately_abnormal": (200, 399),
                "severely_abnormal": (0, 199)
            }
        },
        "active": {
            "sdnn": {
                "normal": (120, 200),
                "mildly_abnormal": (100, 119),
                "moderately_abnormal": (80, 99),
                "severely_abnormal": (0, 79)
            },
            "rmssd": {
                "normal": (200, 500),
                "mildly_abnormal": (150, 199),
                "moderately_abnormal": (100, 149),
                "severely_abnormal": (0, 99)
            },
            "lf_hf": {
                "normal": (0.5, 1.2),
                "mildly_abnormal": (1.3, 1.8),
                "moderately_abnormal": (1.9, 2.5),
                "severely_abnormal": (2.6, float('inf'))
            },
            "lf": {
                "normal": (500, 1400),
                "mildly_abnormal": (350, 499),
                "moderately_abnormal": (200, 349),
                "severely_abnormal": (0, 199)
            },
            "hf": {
                "normal": (600, 1600),
                "mildly_abnormal": (400, 599),
                "moderately_abnormal": (200, 399),
                "severely_abnormal": (0, 199)
            }
        }
    },
    "older_adult": {
        "resting": {
            "sdnn": {
                "normal": (20, 45),
                "mildly_abnormal": (15, 19),
                "moderately_abnormal": (10, 14),
                "severely_abnormal": (0, 9)
            },
            "rmssd": {
                "normal": (15, 30),
                "mildly_abnormal": (10, 14),
                "moderately_abnormal": (6, 9),
                "severely_abnormal": (0, 5)
            },
            "lf_hf": {
                "normal": (1.5, 3.5),
                "mildly_abnormal": (3.6, 4.5),
                "moderately_abnormal": (4.6, 6.0),
                "severely_abnormal": (6.1, float('inf'))
            },
            "lf": {
                "normal": (150, 400),
                "mildly_abnormal": (100, 149),
                "moderately_abnormal": (50, 99),
                "severely_abnormal": (0, 49)
            },
            "hf": {
                "normal": (100, 300),
                "mildly_abnormal": (50, 99),
                "moderately_abnormal": (30, 49),
                "severely_abnormal": (0, 29)
            }
        },
        "active": {
            "sdnn": {
                "normal": (80, 150),
                "mildly_abnormal": (60, 79),
                "moderately_abnormal": (40, 59),
                "severely_abnormal": (0, 39)
            },
            "rmssd": {
                "normal": (100, 250),
                "mildly_abnormal": (70, 99),
                "moderately_abnormal": (50, 69),
                "severely_abnormal": (0, 49)
            },
            "lf_hf": {
                "normal": (1.5, 3.5),
                "mildly_abnormal": (3.6, 4.5),
                "moderately_abnormal": (4.6, 6.0),
                "severely_abnormal": (6.1, float('inf'))
            },
            "lf": {
                "normal": (150, 400),
                "mildly_abnormal": (100, 149),
                "moderately_abnormal": (50, 99),
                "severely_abnormal": (0, 49)
            },
            "hf": {
                "normal": (100, 300),
                "mildly_abnormal": (50, 99),
                "moderately_abnormal": (30, 49),
                "severely_abnormal": (0, 29)
            }
        }
    }
}

BP_ANOMALY_WEIGHTS = {
    "Crisis": 3,
    "Stage2": 2,
    "Stage1": 1,
    "Elevated": 0.5,
    "Moderate": 1.5,
    "Mild": 1.0,
    "Significant": 0.5  # Added for slope-based trends
}

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64)
        c0 = torch.zeros(1, x.size(0), 64)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model_stress = LSTMModel()
model_stress.eval()


def lstm_based_detection(history):
    if len(history) < LSTM_WINDOW_SIZE:
        return []
    window = list(history)[-LSTM_WINDOW_SIZE:]
    input_data = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model_stress(input_data)
    prediction = output.item()
    if prediction > 140:
        return [("Stage1 Predicted", 0.9)]
    elif prediction > 130:
        return [("Elevated Predicted", 0.7)]
    return [(f"Predicted BP score: {prediction:.2f}", 0.5)]

def calculate_anomaly_score(anomaly, lstm_output, category="adult"):
    for key, weight in BP_ANOMALY_WEIGHTS.items():
        if key.lower() in anomaly.lower():
            return weight
    if lstm_output and "Predicted BP score" in lstm_output[0]:
        bp_score = float(lstm_output[0].split(": ")[1])
        threshold = BP_THRESHOLDS.get(category, {}).get("stage1", ((130, 139), (0, 0)))[0][0]
        if bp_score > threshold:
            return 0.5
    return 0.5

def find_consensus(rule_anomalies, lstm_anomalies):
    consensus = []
    rule_texts = [r.lower() for r in rule_anomalies]
    lstm_texts = [l[0].lower() if isinstance(l, tuple) else l.lower() for l in lstm_anomalies]
    for r in rule_texts:
        for l in lstm_texts:
            for stage in ["crisis", "stage2", "stage1", "elevated"]:
                if stage in r and stage in l:
                    consensus.append(r.title())
    return list(set(consensus))

def compute_slope(history, index):
    if len(history) < 3:
        return 0
    values = [item[index] for item in history]
    x = np.arange(len(values))
    y = np.array(values)
    slope, _ = np.polyfit(x, y, 1)
    return slope

def hybrid_decision(rule_anomalies, lstm_output, category):
    score = 0
    all_anomalies = []
    insights = []

    for a in rule_anomalies:
        score += calculate_anomaly_score(a, lstm_output, category)
        all_anomalies.append(a)

    for a in lstm_output:
        if isinstance(a, tuple):
            label, confidence = a
            score += calculate_anomaly_score(label, lstm_output, category) * confidence
            all_anomalies.append(label)
        else:
            score += calculate_anomaly_score(a, lstm_output, category)
            all_anomalies.append(a)

    slope_anomalies = []
    sys_slope = compute_slope(BP_HISTORY, 0)
    dia_slope = compute_slope(BP_HISTORY, 1)

    if sys_slope > 2:
        slope_anomalies.append("Significant Systolic Trend")
        score += BP_ANOMALY_WEIGHTS["Significant"]
    elif sys_slope < -2:
        slope_anomalies.append("Significant Systolic Drop")
        score += BP_ANOMALY_WEIGHTS["Significant"]

    if dia_slope > 2:
        slope_anomalies.append("Significant Diastolic Trend")
        score += BP_ANOMALY_WEIGHTS["Significant"]
    elif dia_slope < -2:
        slope_anomalies.append("Significant Diastolic Drop")
        score += BP_ANOMALY_WEIGHTS["Significant"]

    all_anomalies.extend(slope_anomalies)

    consensus = find_consensus(rule_anomalies, lstm_output)
    if consensus:
        insights.append(f"✅ Consensus found: {', '.join(consensus)}")
        score += 0.5 * len(consensus)
    else:
        insights.append("⚠️ No consensus between rule-based and LSTM outputs.")

    if score >= 4:
        alert = "Stage 3: High Alert – Hypertensive Crisis"
    elif score >= 2.5:
        alert = "Stage 2: Medium Alert – Stage 2 Hypertension"
    elif score >= 1:
        alert = "Stage 1: Soft Alert – Stage 1 Hypertension"
    elif score > 0:
        alert = "Note: Elevated BP – Monitor"
    else:
        alert = "Normal"

    return all_anomalies, alert, score, insights

def classify_bp(systolic: int, diastolic: int, group: str = "adult") -> str:
    thresholds = BP_THRESHOLDS.get(group.lower())
    if thresholds is None:
        raise ValueError(f"Unknown group '{group}'. Must be one of: {list(BP_THRESHOLDS.keys())}")

    priority_order = ['crisis', 'stage2', 'stage1', 'elevated', 'normal']
    for level in priority_order:
        sys_range, dia_range = thresholds[level]
        if sys_range[0] <= systolic <= sys_range[1] or dia_range[0] <= diastolic <= dia_range[1]:
            return level
    return "unclassified"

st.set_page_config(
    page_title="Triksha Health Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div {
        color: white !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #333333;
    }
    div[data-baseweb="option"] {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #2c41c3;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .css-1cpxqw2 {
        color: white !important;
    }
    [data-testid="stSidebar"] .css-1d391kg {
        color: white !important;
        background-color: black !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: white !important;
    }
    div[data-baseweb="option"] {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def status_line(msg, level):
    colors = {
        "ok": "#1f77b4",
        "warning": "#ff7f0e",
        "critical": "#d62728",
    }
    color = colors[level]
    st.markdown(f"<span style='color:{color}; font-weight:bold'>{msg}</span>", unsafe_allow_html=True)

def altair_chart_from_data(df, name):
    df['Index'] = df.index
    if name == "Blood Pressure":
        chart = alt.Chart(df).transform_fold(
            ["Systolic", "Diastolic"],
            as_=['Type', 'Value']
        ).mark_line().encode(
            x=alt.X('Index:Q', title='Time'),
            y=alt.Y('Value:Q', title='Blood Pressure'),
            color=alt.Color('Type:N', scale=alt.Scale(domain=['Systolic', 'Diastolic'], range=['#1f77b4', '#ff7f0e']))
        )
    else:
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('Index:Q', title='Time'),
            y=alt.Y(f'{name}:Q', title=name),
            color=alt.value('#1f77b4')
        )
    return chart.configure_axis(labelColor='black', titleColor='black').configure_view(strokeWidth=0)

def moving_average(history, idx):
    vals = [entry[idx] for entry in history]
    return sum(vals) / len(vals) if vals else 0

def compute_std(history, idx):
    avg = moving_average(history, idx)
    vals = [entry[idx] for entry in history]
    if not vals:
        return 0
    variance = sum((x - avg) ** 2 for x in vals) / len(vals)
    return math.sqrt(variance)

def compute_slope(history, idx):
    if len(history) < 5:
        return 0
    y = [entry[idx] for entry in history]
    x = list(range(len(y)))
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator != 0 else 0

def check_sudden_bp_change(systolic, diastolic):
    anomalies = []
    if len(BP_HISTORY) >= 5:
        mean_sys = moving_average(BP_HISTORY, 0)
        std_sys = compute_std(BP_HISTORY, 0)
        if std_sys > 0:
            z_sys = (systolic - mean_sys) / std_sys
            if abs(z_sys) > 2:
                anomalies.append("Moderate - Sudden systolic BP change")
        mean_dia = moving_average(BP_HISTORY, 1)
        std_dia = compute_std(BP_HISTORY, 1)
        if std_dia > 0:
            z_dia = (diastolic - mean_dia) / std_dia
            if abs(z_dia) > 2:
                anomalies.append("Moderate - Sudden diastolic BP change")
    return anomalies

def check_bp_pattern_anomalies():
    if len(BP_HISTORY) < 5:
        return []
    anomalies = []
    sys_vals = [x[0] for x in BP_HISTORY]
    dia_vals = [x[1] for x in BP_HISTORY]
    if max(sys_vals) - min(sys_vals) > 20:
        anomalies.append("Mild - Systolic BP fluctuation detected")
    if max(dia_vals) - min(dia_vals) > 15:
        anomalies.append("Mild - Diastolic BP fluctuation detected")
    sys_slope = compute_slope(BP_HISTORY, 0)
    dia_slope = compute_slope(BP_HISTORY, 1)
    if sys_slope > 1:
        anomalies.append("Mild - Gradual systolic BP rise")
    elif sys_slope < -1:
        anomalies.append("Mild - Gradual systolic BP drop")
    if dia_slope > 1:
        anomalies.append("Mild - Gradual diastolic BP rise")
    elif dia_slope < -1:
        anomalies.append("Mild - Gradual diastolic BP drop")
    return anomalies

def check_sudden_temp_change(temperature):
    anomalies = []
    if len(TEMP_HISTORY) >= 5:
        mean_temp = moving_average(TEMP_HISTORY)
        std_temp = compute_std(TEMP_HISTORY)
        if std_temp > 0:
            z_temp = (temperature - mean_temp) / std_temp
            if abs(z_temp) > 2:
                anomalies.append("Moderate - Sudden temperature change")
    TEMP_HISTORY.append(temperature)
    TEMP_SEQUENCE.append(temperature)
    return anomalies

def check_temp_pattern_anomalies():
    if len(TEMP_HISTORY) < 5:
        return []
    anomalies = []
    temp_vals = list(TEMP_HISTORY)

    if all(temp_vals[i] >= temp_vals[i+1] for i in range(len(temp_vals)-1)):
        anomalies.append("Mild - Gradual temperature drop")
    if max(temp_vals) - min(temp_vals) > 1.5:
        anomalies.append("Mild - Temperature fluctuation detected")

    return anomalies


def rule_based_detection(data):
    anomalies = []
    systolic = data['systolic']
    diastolic = data['diastolic']
    category = data['category']
    bp_stage = classify_bp(systolic, diastolic, category)
    if bp_stage != "Normal":
        anomalies.append(bp_stage)
    anomalies += check_sudden_bp_change(systolic, diastolic)
    anomalies += check_bp_pattern_anomalies()
    return anomalies

def simulate_vital(name, unit, lo, hi, threshold=None):
    data = []
    chart = st.empty()
    info = st.empty()
    v_input = st.number_input(
        f"Manual {name} input ({unit.strip()})",
        min_value=float(lo),
        max_value=float(hi),
        value=round((lo + hi) / 2, 2),
        step=0.1
    )
    if st.button("Start Simulation"):
        for _ in range(50):
            v = round(random.uniform(lo, hi), 2)
            data.append(v)
            df = pd.DataFrame(data, columns=[name])
            chart.altair_chart(altair_chart_from_data(df, name), use_container_width=True)
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
            time.sleep(0.5)

class EnhancedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EnhancedLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_lstm = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x, h_prev, c_prev, forget_mask):
        combined = torch.cat((x, h_prev), dim=1)
        gates = self.W_lstm(combined)
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t) * forget_mask.unsqueeze(1)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)
        c_t = f_t * c_prev + i_t * g_t
        c_t = self.ln(c_t)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class Temp_LSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, attention_size=32, output_size=5, bidirectional=False):
        super(Temp_LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm_cell = EnhancedLSTMCell(input_size, hidden_size)
        if bidirectional:
            self.lstm_cell_backward = EnhancedLSTMCell(input_size, hidden_size)
        self.W_query = nn.Linear(hidden_size, attention_size)
        self.W_key = nn.Linear(hidden_size, attention_size)
        self.W_value = nn.Linear(hidden_size, attention_size)
        self.fc = nn.Linear(attention_size, output_size)
        self.ln = nn.LayerNorm(attention_size)

    def forward(self, x, abnormal_flags):
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        hidden_states = []

        for t in range(seq_len):
            forget_mask = abnormal_flags[:, t].to(x.device)
            h_t, c_t = self.lstm_cell(x[:, t, :], h_t, c_t, forget_mask)
            hidden_states.append(h_t.unsqueeze(1))
        
        if self.bidirectional:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            hidden_states_backward = []
            for t in range(seq_len-1, -1, -1):
                forget_mask = abnormal_flags[:, t].to(x.device)
                h_t, c_t = self.lstm_cell_backward(x[:, t, :], h_t, c_t, forget_mask)
                hidden_states_backward.insert(0, h_t.unsqueeze(1))
            hidden_states = [torch.cat((hf, hb), dim=-1) for hf, hb in zip(hidden_states, hidden_states_backward)]

        hidden_states = torch.cat(hidden_states, dim=1)
        queries = self.W_query(hidden_states)
        keys = self.W_key(hidden_states)
        values = self.W_value(hidden_states)
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attn_scores = attn_scores + abnormal_flags.unsqueeze(1).to(x.device) * 1.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vector = torch.bmm(attn_weights, values)
        context_vector = self.ln(context_vector[:, -1, :])
        output = self.fc(context_vector)
        return output, attn_weights[:, -1, :]

# Initialize LSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temp_lstm = Temp_LSTM_Model().to(device)

# ==================== Decision Layer ====================
class TempDecisionLayer:
    def __init__(self):
        # Severity weights for different anomaly types
        self.severity_weights = {
            "critical": 3.5,
            "Severe": 3.0,
            "Moderate": 2.0,
            "Mild": 1.0,
            "Pattern": 0.8,  # For pattern-based anomalies
            "Elevated": 0.5
        }

        # Alert thresholds
        self.alert_thresholds = {
            "CRITICAL": 4.0,
            "HIGH": 3.0,
            "MEDIUM": 2.0,
            "LOW": 1.0
        }

    def _calculate_anomaly_score(self, anomaly):
        """Calculate score for individual anomaly based on severity"""
        if not isinstance(anomaly, str):
            return 0.0

        for severity, weight in self.severity_weights.items():
            if severity.lower() in anomaly.lower():
                return weight
        return 0.0

    def _find_consensus_anomalies(self, rule_anomalies, lstm_anomalies):
        """Find anomalies where both models agree"""
        consensus = []

        # Convert to sets for faster lookup
        rule_set = set(str(a).lower() for a in rule_anomalies)
        lstm_set = set(str(a).lower() for a in lstm_anomalies)

        # Find matching severity levels
        for r_anom in rule_anomalies:
            if not isinstance(r_anom, str):
                continue

            r_severity = next((s for s in self.severity_weights
                             if s.lower() in r_anom.lower()), None)

            for l_anom in lstm_anomalies:
                if not isinstance(l_anom, str):
                    continue

                l_severity = next((s for s in self.severity_weights
                                  if s.lower() in l_anom.lower()), None)

                # Match if same severity found
                if r_severity and l_severity and r_severity == l_severity:
                    consensus.append(r_anom)
                    break

        return consensus

class EnhancedLSTMWithAttentionBP(nn.Module):
    def __init__(self, input_size, hidden_size, attention_size, output_size, bidirectional=False):
        super(EnhancedLSTMWithAttentionBP, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm_cell = EnhancedLSTMCell(input_size, hidden_size)
        if bidirectional:
            self.lstm_cell_backward = EnhancedLSTMCell(input_size, hidden_size)
        self.W_query = nn.Linear(hidden_size * self.num_directions, attention_size)
        self.W_key = nn.Linear(hidden_size * self.num_directions, attention_size)
        self.W_value = nn.Linear(hidden_size * self.num_directions, attention_size)
        self.fc = nn.Linear(attention_size, output_size)
        self.ln = nn.LayerNorm(attention_size)

    def forward(self, x, abnormal_flags):
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        hidden_states = []
        for t in range(seq_len):
            forget_mask = abnormal_flags[:, t].to(x.device)
            h_t, c_t = self.lstm_cell(x[:, t, :], h_t, c_t, forget_mask)
            hidden_states.append(h_t.unsqueeze(1))
        if self.bidirectional:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            hidden_states_backward = []
            for t in range(seq_len - 1, -1, -1):
                forget_mask = abnormal_flags[:, t].to(x.device)
                h_t, c_t = self.lstm_cell_backward(x[:, t, :], h_t, c_t, forget_mask)
                hidden_states_backward.insert(0, h_t.unsqueeze(1))
            hidden_states = [torch.cat((hf, hb), dim=-1) for hf, hb in zip(hidden_states, hidden_states_backward)]
        hidden_states = torch.cat(hidden_states, dim=1)
        queries = self.W_query(hidden_states)
        keys = self.W_key(hidden_states)
        values = self.W_value(hidden_states)
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attn_scores = attn_scores + abnormal_flags.unsqueeze(1).to(x.device) * 1.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vector = torch.bmm(attn_weights, values)
        context_vector = self.ln(context_vector[:, -1, :])
        output = self.fc(context_vector)
        return output, attn_weights[:, -1, :]

def lstm_detection(bp_history):
    if len(bp_history) < LSTM_WINDOW_SIZE:
        return []
    inputs = np.array(bp_history[-LSTM_WINDOW_SIZE:])
    inputs = torch.FloatTensor(inputs).unsqueeze(0).to(device)
    abnormal_flags = torch.zeros(1, LSTM_WINDOW_SIZE).to(device)
    with torch.no_grad():
        output, _ = model_bp(inputs, abnormal_flags)
        prediction = output.item()
        confidence = min(1.0, max(0.0, prediction / 180))  # Normalize over expected max
    if prediction > 140:
        return [(f"Predicted High BP Score: {prediction:.2f}", confidence)]
    return []
def classify_temp(temperature, category):
    thresholds = TEMP_THRESHOLDS.get(category, {})
    for label, temp_range in thresholds.items():
        if temp_range[0] <= temperature <= temp_range[1]:
            return label.title().replace("_", " ")
    return "Unclassified"

def detect_temp_abnormalities(data):
    temperature = data['temperature']
    category = data['category']
    anomalies = []

    temp_level = classify_temp(temperature, category)
    if temp_level != "Normal":
        anomalies.append(temp_level)

    anomalies += check_sudden_temp_change(temperature)
    anomalies += check_temp_pattern_anomalies()
    slope = slope_based_trend_score(TEMP_HISTORY)

    return anomalies, slope

def detect_lstm_temp_anomalies():
    if len(TEMP_SEQUENCE) < 10:
        return "Normal: Insufficient data"

    seq_data = torch.tensor(list(TEMP_SEQUENCE)).unsqueeze(0).unsqueeze(-1).float().to(device)
    abnormal_flags = torch.zeros(1, len(TEMP_SEQUENCE))

    with torch.no_grad():
        output, _ = temp_lstm(seq_data, abnormal_flags.to(device))
        pred_class = torch.argmax(output, dim=1).item()

    class_labels = ["Normal", "Mild", "Moderate", "Severe", "Critical"]
    return f"{class_labels[pred_class]}: LSTM prediction"

decision_layer = TempDecisionLayer()
def home():
    st.markdown('<div class="blue-title">🏥 Health Insights Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Welcome to Triksha")
    st.write("AI-powered real-time vital monitoring.")
    st.write("---")
    age = st.slider("Age", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    athlete_status = st.selectbox("Athlete Status", ["Athlete", "Non-Athlete"])
    category = st.selectbox("Category", ["Diabetic", "Cardiac", "Respiratory", "General"])
    st.session_state["bp_category"] = "adult" if category == "Cardiac" else "adult"
    activity = st.selectbox("Activity Level", ["Resting", "Active"])
    st.info("👈 Use the sidebar to explore vitals.")

def heart(): st.title("❤️ Heart Rate"); simulate_vital("Heart Rate", " bpm", 60, 130, 100)

def hrv(): st.title("💓 HRV"); simulate_vital("HR Variability", " ms", 20, 120, 80)

def resp(): st.title("🌬️ Resp Rate"); simulate_vital("Respiratory Rate", " br/min", 12, 25, 20)

def spo2(): st.title("🦢 SpO₂"); simulate_vital("SpO2", " %", 90, 100, 94)

def bp():
    st.title("🩸 Blood Pressure")
    sys_list, dia_list = [], []
    ch = st.empty()
    inf = st.empty()
    explanation = st.empty()

    if "bp_category" not in st.session_state:
        st.session_state["bp_category"] = "adult"

    if st.button("Start BP Simulation"):
        for _ in range(50):
            systolic = round(random.uniform(100, 180), 2)
            diastolic = round(random.uniform(60, 100), 2)
            BP_HISTORY.append((systolic, diastolic))
            sys_list.append(systolic)
            dia_list.append(diastolic)

            selected_category = st.session_state.get("bp_category", "adult")
            rule_inputs = {
                "systolic": systolic,
                "diastolic": diastolic,
                "category": selected_category
            }
            rule_anomalies = rule_based_detection(rule_inputs)

            try:
                lstm_output = lstm_detection(BP_HISTORY)
            except Exception as e:
                st.error(f"LSTM detection failed: {str(e)}")
                lstm_output = []

            anomalies, alert, score, insights = hybrid_decision(rule_anomalies, lstm_output, selected_category)

            df = pd.DataFrame({"Systolic": sys_list, "Diastolic": dia_list})
            df['Index'] = df.index
            chart = alt.Chart(df).transform_fold(
                ["Systolic", "Diastolic"],
                as_=['Type', 'Value']
            ).mark_line().encode(
                x=alt.X('Index:Q', title='Time'),
                y=alt.Y('Value:Q', title='Blood Pressure'),
                color=alt.Color('Type:N', scale=alt.Scale(domain=['Systolic', 'Diastolic'], range=['#1f77b4', '#ff7f0e']))
            ).configure_axis(
                labelColor='black',
                titleColor='black'
            ).configure_view(strokeWidth=0)
            ch.altair_chart(chart, use_container_width=True)

            with inf.container():
                st.subheader("🔍 Insights")
                all_anomalies = [a.lower() for a in rule_anomalies]
                if "crisis" in all_anomalies:
                    status_line(f"🚨 CRITICAL: {systolic}/{diastolic}", "critical")
                elif "stage2" in all_anomalies:
                    status_line(f"⚠️ Stage 2: {systolic}/{diastolic}", "critical")
                elif "stage1" in all_anomalies:
                    status_line(f"🟠 Stage 1: {systolic}/{diastolic}", "warning")
                elif "elevated" in all_anomalies:
                    status_line(f"⬆️ Elevated: {systolic}/{diastolic}", "warning")
                else:
                    status_line(f"✔️ Normal: {systolic}/{diastolic}", "ok")

                if lstm_output:
                    pred, conf = lstm_output[0]
                    st.write(f"🧠 LSTM Model: `{pred}` | Confidence: `{conf * 100:.1f}%`")

                all_anomalies_output = rule_anomalies + check_sudden_bp_change(systolic, diastolic) + check_bp_pattern_anomalies()
                if all_anomalies_output:
                    st.write("• Detected anomalies:")
                    for a in all_anomalies_output:
                        st.markdown(f"- {a}")
                else:
                    st.write("• No anomalies detected.")

            with explanation.expander("ℹ️ What does this mean?"):
                st.markdown("""
                    - **Normal:** BP within safe range.
                    - **Elevated:** Early warning.
                    - **Stage 1/2:** Hypertension risk.
                    - **Crisis:** Medical emergency.
                    - **Z-Score/Pattern:** Detects sudden or gradual changes.
                    - **LSTM:** AI-predicted risk patterns.
                """)

            time.sleep(0.25)

def glucose(): st.title("🧃 Glucose"); simulate_vital("Glucose", " mg/dL", 70, 200, 140)

def temp(): st.title("🌡️ Temperature"); simulate_vital("Body Temperature", " °C", 36, 39.5, 37.5)

def stress(): st.title("😰 Stress"); simulate_vital("Stress Level", "/10", 1, 10, 7)

def sleep(): st.title("🛌 Sleep"); simulate_vital("Sleep Duration", " hrs", 0, 10, 5)

PAGES = {
    "Home": home, "Heart Rate": heart, "HRV": hrv,
    "Resp Rate": resp, "SpO2": spo2, "Blood Pressure": bp,
    "Glucose": glucose, "Temperature": temp,
    "Stress": stress, "Sleep": sleep
}

st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox("Go to", list(PAGES.keys()))
PAGES[page]()
