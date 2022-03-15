# With help from 6.07 lesson
import streamlit as st
import numpy as np
import pickle

st.title("Pipeline Score Predictor")
st.header("Let's predict the average heat score at pipeline right now!")

# Define features with '0' coefficients from lasso
wave_height = 0
dominant_period = 0
avg_period = 0
dominant_wave_direction_sin = 0
wave_height_1_h = 0
dominant_period_1_h = 0
dominant_wave_direction_sin_1_h = 0
dominant_wave_direction_cos_1_0_h = 0
wave_height_1_5_h = 0
avg_period_1_5_h = 0
dominant_wave_direction_sin_1_5_h = 0
dominant_wave_direction_cos_1_5_h = 0
dominant_period_3_h = 0
avg_period_3_h = 0
dominant_wave_direction_sin_3_h = 0
dominant_wave_direction_cos_3_h = 0
wave_height_6_h = 0
dominant_wave_direction_sin_6_h = 0
wave_height_9_h = 0
dominant_period_9_h = 0
dominant_wave_direction_sin_9_h = 0
dominant_wave_direction_cos_9_h = 0
gust_speed = 0

# Get user input for lasso features
avg_period_9_h = st.number_input("Average Wave Period 9 hours ago", 0., 20.)
avg_period_6_h = st.number_input("Average Wave Period 6 hours ago", 0., 20.)

# Adjust wind direction
wind_direction = st.number_input("Current Wind Direction", 0, 359)
wind_direction_cos = np.cos(wind_direction * (2.0 * np.pi / 360))
wind_direction_sin = np.sin(wind_direction * (2.0 * np.pi / 360))

# Adjust wave direction
dominant_wave_direction = st.number_input("Current Dominant Wave Direction", 0, 359)
dominant_wave_direction_cos = np.cos(dominant_wave_direction * (2.0 * np.pi / 360))
dominant_wave_direction_6_h = st.number_input(
    "Dominant Wave Direction 6 hours ago", 0, 359
)
dominant_wave_direction_cos_6_h = np.cos(
    dominant_wave_direction_6_h * (2.0 * np.pi / 360)
)

avg_period_1_h = st.number_input("Average Wave Period 1 hour ago", 0., 20.)
wind_speed = st.number_input("Current Wind Speed", 0., 50.)


dominant_period_1_5_h = st.number_input("Dominant Wave Period 1.5 hours ago", 0., 20.)
dominant_period_6_h = st.number_input("Dominant Wave Period 6 hours ago", 0., 20.)
wave_height_3_h = st.number_input("Wave Height 3 hours ago", 0., 20.)

user_input = np.array(
    [
        wave_height,
        dominant_period,
        avg_period,
        dominant_wave_direction_sin,
        dominant_wave_direction_cos,
        wave_height_1_h,
        dominant_period_1_h,
        avg_period_1_h,
        dominant_wave_direction_sin_1_h,
        dominant_wave_direction_cos_1_0_h,
        wave_height_1_5_h,
        dominant_period_1_5_h,
        avg_period_1_5_h,
        dominant_wave_direction_sin_1_5_h,
        dominant_wave_direction_cos_1_5_h,
        wave_height_3_h,
        dominant_period_3_h,
        avg_period_3_h,
        dominant_wave_direction_sin_3_h,
        dominant_wave_direction_cos_3_h,
        wave_height_6_h,
        dominant_period_6_h,
        avg_period_6_h,
        dominant_wave_direction_sin_6_h,
        dominant_wave_direction_cos_6_h,
        wave_height_9_h,
        dominant_period_9_h,
        avg_period_9_h,
        dominant_wave_direction_sin_9_h,
        dominant_wave_direction_cos_9_h,
        wind_speed,
        gust_speed,
        wind_direction_sin,
        wind_direction_cos,
    ]
).reshape(1, -1)

# Context block to open file and load
with open("../saved-models/saved-lasso.pkl", "rb") as f:
    regressor = pickle.load(f)

# prediction
prediction = regressor.predict(user_input)

check = st.button("I'm ready for a prediction!")

if check:
    st.header(f"The model predicts: {round(prediction[0], 2)}!")