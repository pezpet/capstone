# With help from 6.07 lesson
import streamlit as st
import numpy as np
import pickle
from requests import get
from bs4 import BeautifulSoup

# Get buoy data
url = "https://www.ndbc.noaa.gov/station_page.php?station=51201"
res = get(url)
soup = BeautifulSoup(res.content, features="lxml")

# Get wind data
url = "https://www.ndbc.noaa.gov/station_page.php?station=oouh1"
res = get(url)
soup2 = BeautifulSoup(res.content, features="lxml")

# Cardinal directions mapper
directions = {
    "N": 0.,
    "NNE": 23.,
    "NE": 45.,
    "NEE": 68.,
    "E": 90.,
    "SEE": 113.,
    "SE": 135.,
    "SSE": 158.,
    "S": 180.,
    "SSW": 203.,
    "SW": 225.,
    "SWW": 248.,
    "W": 270.,
    "NWW": 293.,
    "NW": 315.,
    "NNW": 338.,
}

# Assign features with '0' coefficients from lasso
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
# Average period 9 hours ago
avg_period_9_h = float(
    soup.find("table", class_="dataTable").find_all("tr")[19].find_all("td")[8].text
)

# Average period 6 hours ago
avg_period_6_h = float(
    soup.find("table", class_="dataTable").find_all("tr")[13].find_all("td")[8].text
)

# Current Wind Direction
wind_direction = float(
    soup2.find("div", id="data").find_all("tr")[1].find_all("td")[2].text.split()[2]
)

# Adjust wind directions to sin and cos
wind_direction_cos = np.cos(wind_direction * (2.0 * np.pi / 360))
wind_direction_sin = np.sin(wind_direction * (2.0 * np.pi / 360))

# Current wave direction
dominant_wave_direction = float(
    soup.find("div", id="data")
    .find_all("tr")[4]
    .find_all("td")[2]
    .text.strip()
    .split()[2]
)

# Adjust wave directions to cos
dominant_wave_direction_cos = np.cos(dominant_wave_direction * (2.0 * np.pi / 360))

# Wave direction 6 hours ago
dominant_wave_direction_6_h = directions[
    soup.find("table", class_="dataTable").find_all("tr")[13].find_all("td")[9].text
]

# Adjust wave directions to cos
dominant_wave_direction_cos_6_h = np.cos(
    dominant_wave_direction_6_h * (2.0 * np.pi / 360)
)

# Average wave period 1 hour ago
avg_period_1_h = float(
    soup.find("table", class_="dataTable").find_all("tr")[3].find_all("td")[8].text
)

# Current Wind Speed
wind_speed = float(
    soup2.find("div", id="data").find_all("tr")[2].find_all("td")[2].text.split()[0]
)

# Dominant Wave Period 1.5 hours ago
dominant_period_1_5_h = float(
    soup.find("table", class_="dataTable").find_all("tr")[4].find_all("td")[7].text
)

# Dominant Wave Period 6 hours ago
dominant_period_6_h = float(
    soup.find("table", class_="dataTable").find_all("tr")[13].find_all("td")[7].text
)

# Wave Height 3 hours ago
wave_height_3_h = float(
    soup.find("table", class_="dataTable").find_all("tr")[7].find_all("td")[6].text
)

# Streamlit
st.title("Pipeline Score Predictor")
st.header("Let's predict the average heat score at pipeline right now!")
st.header("This is the data from NOAA:")
st.subheader("Wind Station: https://www.ndbc.noaa.gov/station_page.php?station=oouh1")
st.text(f"Current Wind Speed: {wind_speed}")
st.text(f"Current Wind Direction: {wind_direction}")

st.subheader("Buoy: https://www.ndbc.noaa.gov/station_page.php?station=51201")
st.text(f"Current Wave Direction: {dominant_wave_direction}")
st.text(f"Avg. Wave Period 1 hour ago: {avg_period_1_h}")
st.text(f"Dominant Wave Period 1.5 hours ago: {dominant_period_1_5_h}")
st.text(f"Wave Height 3 hours ago: {wave_height_3_h}")
st.text(f"Wave Direction 6 hours ago: {dominant_wave_direction_6_h}")
st.text(f"Dominant Wave Period 6 hours ago: {dominant_period_6_h}")
st.text(f"Average Wave Period 6 hours ago: {avg_period_6_h}")
st.text(f"Average Wave Period 9 hours ago: {avg_period_9_h}")

model_input = np.array(
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
prediction = regressor.predict(model_input)

st.header(f"The model predicts: {round(prediction[0], 2)}!")
