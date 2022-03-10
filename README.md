# Executive Summary - Predicting World Surf League Scores
## Goals
The goal of this project is to use raw buoy data to predict scores in surfing competition heats. The WSL uses sofisticated forecasting models to estimate the quality of waves at each location were events are held, and even though "scoring potential" is frequently mentioned as a critical variable, no specific modeling or forecasting is described when deciding competition dates and times. By creating a model that targets scores, this projects aims to improve the decision making process of the WSL.

## Data collection
### World Surf League
The World Surf League organizes a yearly world tour of surfing competitions and is regarded as the highest level of competitive surfing. Their website keeps scores for all competitions back to 2008. These scores were scraped from the site. Additionally, to be able to link scores to buoy data, the dates for all heats and all events were also scraped.

### NOAA (National Buoy Data Center)
The National Buoy Data Center (part of NOAA) keeps historic data from marine buoys around the world for oceanic conditions, including wave height, direction, period, and other information. This historic data is available for download in txt format by year by buoy.



WDIR| Wind direction (the direction the wind is coming from in degrees clockwise from true N) during the same period used for WSPD. See Wind Averaging Methods
WSPD	Wind speed (m/s) averaged over an eight-minute period for buoys and a two-minute period for land stations. Reported Hourly. See Wind Averaging Methods.
GST	Peak 5 or 8 second gust speed (m/s) measured during the eight-minute or two-minute period. The 5 or 8 second period can be determined by payload, See the Sensor Reporting, Sampling, and Accuracy section.
WVHT	Significant wave height (meters) is calculated as the average of the highest one-third of all of the wave heights during the 20-minute sampling period. See the Wave Measurements section.
DPD	Dominant wave period (seconds) is the period with the maximum wave energy. See the Wave Measurements section.
APD	Average wave period (seconds) of all waves during the 20-minute period. See the Wave Measurements section.
MWD	The direction from which the waves at the dominant period (DPD) are coming. The units are degrees from true North, increasing clockwise, with North as 0 (zero) degrees and East as 90 degrees. See the Wave Measurements section.
PRES	Sea level pressure (hPa). For C-MAN sites and Great Lakes buoys, the recorded pressure is reduced to sea level using the method described in NWS Technical Procedures Bulletin 291 (11/14/80). ( labeled BAR in Historical files)
ATMP	Air temperature (Celsius). For sensor heights on buoys, see Hull Descriptions. For sensor heights at C-MAN stations, see C-MAN Sensor Locations
WTMP	Sea surface temperature (Celsius). For buoys the depth is referenced to the hull's waterline. For fixed platforms it varies with tide, but is referenced to, or near Mean Lower Low Water (MLLW).
DEWP	Dewpoint temperature taken at the same height as the air temperature measurement.
VIS	Station visibility (nautical miles). Note that buoy stations are limited to reports from 0 to 1.6 nmi.
PTDY	Pressure Tendency is the direction (plus or minus) and the amount of pressure change (hPa)for a three hour period ending at the time of observation. (not in Historical files)
TIDE	The water level in feet above or below Mean Lower Low Water (MLLW).