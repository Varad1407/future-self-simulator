import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor

# -------------------------
# TRAIN MODEL (NO model.pkl)
# -------------------------
@st.cache_resource
def train_model():
    data = []

    for _ in range(1000):
        sleep = np.random.randint(3, 10)
        workout = np.random.randint(0, 120)
        work_hours = np.random.randint(0, 12)
        screen_time = np.random.randint(0, 12)
        diet = np.random.randint(1, 10)

        score = (
            (20 if 7 <= sleep <= 8 else 10 if sleep >= 6 else 0) +
            min(workout / 5, 20) +
            min(work_hours * 2, 20) +
            (20 if screen_time <= 2 else 10 if screen_time <= 4 else 0) +
            diet * 2
        )

        data.append([sleep, workout, work_hours, screen_time, diet, score])

    df = pd.DataFrame(data, columns=[
        "sleep", "workout", "work_hours", "screen_time", "diet", "score"
    ])

    X = df.drop("score", axis=1)
    y = df["score"]

    model = RandomForestRegressor()
    model.fit(X, y)

    return model


model = train_model()

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Future Self Simulator", layout="wide")

st.title("🔮 Future Self Simulator")
st.markdown("### Predict your future based on your daily habits")

# -------------------------
# INPUTS (SIDEBAR)
# -------------------------
st.sidebar.header("Enter Your Daily Habits")

sleep = st.sidebar.slider("Sleep (hours)", 0, 10, 6)
workout = st.sidebar.slider("Workout (minutes)", 0, 120, 20)
work_hours = st.sidebar.slider("Work/Study (hours)", 0, 12, 4)
screen_time = st.sidebar.slider("Screen Time (hours)", 0, 12, 6)
diet = st.sidebar.slider("Diet Quality (1-10)", 1, 10, 5)

# -------------------------
# SIMULATION FUNCTION
# -------------------------
def simulate(score, days):
    growth = score * days * 0.05

    if growth > 600:
        return "🚀 Massive Transformation"
    elif growth > 300:
        return "💪 Strong Progress"
    elif growth > 150:
        return "📈 Moderate Improvement"
    else:
        return "⚠️ Stagnation"

# -------------------------
# BUTTON
# -------------------------
if st.button("Simulate My Future"):

    # ML PREDICTION
    input_data = np.array([[sleep, workout, work_hours, screen_time, diet]])
    current_score = int(model.predict(input_data)[0])

    ideal_input = np.array([[8, max(workout,60), max(work_hours,8), min(screen_time,2), max(diet,8)]])
    ideal_score = int(model.predict(ideal_input)[0])

    # -------------------------
    # SAVE HISTORY
    # -------------------------
    new_data = pd.DataFrame([{
        "sleep": sleep,
        "workout": workout,
        "work_hours": work_hours,
        "screen_time": screen_time,
        "diet": diet,
        "score": current_score
    }])

    if os.path.exists("history.csv"):
        new_data.to_csv("history.csv", mode='a', header=False, index=False)
    else:
        new_data.to_csv("history.csv", index=False)

    # -------------------------
    # SCORE DISPLAY
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Current You")
        st.progress(current_score)
        st.write(f"Score: {current_score}/100")

    with col2:
        st.subheader("🔥 Ideal You")
        st.progress(ideal_score)
        st.write(f"Score: {ideal_score}/100")

    st.divider()

    # -------------------------
    # FUTURE COMPARISON
    # -------------------------
    st.subheader("🔮 Future Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Current Path")
        st.write("30 Days:", simulate(current_score, 30))
        st.write("90 Days:", simulate(current_score, 90))
        st.write("180 Days:", simulate(current_score, 180))

    with col2:
        st.markdown("### Ideal Path")
        st.write("30 Days:", simulate(ideal_score, 30))
        st.write("90 Days:", simulate(ideal_score, 90))
        st.write("180 Days:", simulate(ideal_score, 180))

    st.divider()

    # -------------------------
    # RISK + INSIGHT
    # -------------------------
    col1, col2 = st.columns([1,2])

    with col1:
        st.subheader("⚠️ Risk Level")

        if current_score >= 70:
            st.success("Low Risk")
        elif current_score >= 40:
            st.warning("Medium Risk")
        else:
            st.error("High Risk")

    with col2:
        st.subheader("🧠 Key Insight")

        if current_score < 40:
            st.error("You are on a path of stagnation. Your habits need a complete reset.")
        elif screen_time > 5:
            st.warning("Your biggest issue is screen time. Reduce it immediately.")
        elif workout < 20:
            st.warning("Your physical activity is too low.")
        else:
            st.success("You are on a strong path. Stay consistent.")

    st.divider()

    # -------------------------
    # GRAPH
    # -------------------------
    st.subheader("📊 Growth Comparison")

    days = [30, 90, 180]
    current_values = [current_score*d*0.05 for d in days]
    ideal_values = [ideal_score*d*0.05 for d in days]

    plt.figure()
    plt.plot(days, current_values, marker='o', label="Current You")
    plt.plot(days, ideal_values, marker='o', linestyle='--', label="Ideal You")

    plt.xlabel("Days")
    plt.ylabel("Growth")
    plt.title("Future Projection")
    plt.legend()

    st.pyplot(plt)

    # -------------------------
    # HISTORY DISPLAY
    # -------------------------
    st.subheader("📈 Your Past Data")

    if os.path.exists("history.csv"):
        history = pd.read_csv("history.csv")
        st.dataframe(history.tail(10))