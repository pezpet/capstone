# Executive Summary - Predicting World Surf League Scores
## Goals
The goal of this project is to use raw buoy data to predict scores in surfing competition heats. The WSL uses sofisticated forecasting models to estimate the quality of waves at each location were events are held, and even though "scoring potential" is frequently mentioned as a critical variable, no specific modeling or forecasting is described when deciding competition dates and times. By creating a model that targets scores, this projects aims to improve the decision making process of the WSL.

Different models were built and evaluated on **Root Mean Squared Error (RMSE)**. RMSE on test sets:
- Baseline (RMSE: 3.17)
- Linear Regression with Lasso (RMSE: 2.82)
- Extreme Gradient Boosting (RMSE: 2.87)
- Neural Networks (RMSE: 2.86)

## Data collection
### World Surf League
The World Surf League organizes a yearly world tour of surfing competitions and is regarded as the highest level of competitive surfing. Their website keeps scores for all competitions back to 2008. These scores were scraped from the site. Additionally, to be able to link scores to buoy data, the dates for all heats and all events were also scraped. [Example event page that was scraped](https://www.worldsurfleague.com/events/2008/mct/75/billabong-pipeline-masters)

#### Scores data description:
| Column Name | Data Description | dtype |
|---|---|---|
| year | Year in which the competition was held | ``int`` |
| event | Event Name | `str` |
| round | Competition Round | `str` |
| heat | Competition Heat within Round | `str` |
| name | Athlete Name competing in Heat | `str` |
| top_two_waves_total | Sum of the two highest scoring waves for each athlete in a Heat | `float` |

#### Round Dates data description:
| Column Name | Data Description | dtype |
|---|---|---|
| year | Year in which the competition was held | `int` |
| event | Event Name | `str` |
| round | Competition Round | `str` |
| heat | Competition Heat within Round | `str` |
| date | Date when each Heat ended | `str` |


### NOAA (National Buoy Data Center)
The National Buoy Data Center (part of NOAA) keeps historic data from marine buoys around the world for oceanic conditions, including wave height, direction, period, and other information. This historic data is available for download in txt format by year by buoy. There are also weather stations that log historic wind data.

#### [Waimea Buoy Data Source](https://www.ndbc.noaa.gov/station_page.php?station=51201)

#### [Honolulu Wind Data Source](https://www.ndbc.noaa.gov/station_page.php?station=oouh1)

#### NOAA data description:
| Column Name | Data Description | dtype |
|---|---|---|
| #YY | Year of measurement | `int` |
| MM | Month of measurement | `int` |
| DD | Day of measurement | `int` |
| hh | Hour of measurement | `int` |
| mm | Minute of measurement | `int` |
| WDI | Wind Direction, degrees clockwise from true north | `int` |
| R WSP | Wind Speed, meters per second (m/s) | `int` |
| D GST | Peak 5 or 8 second gust speed (m/s) measured during the eight-minute or two-minute period | `float` |
| WVHT | Significant wave height (meters) is calculated as the average of the highest one-third of all of the wave heights during the 20-minute sampling period. | `float` |
| DPD | Dominant wave period (seconds) is the period with the maximum wave energy. | `float` |
| APD | Average wave period (seconds) of all waves during the 20-minute period. | `float` |
| MWD | The direction from which the waves at the dominant period (DPD) are coming. The units are degrees from true North, increasing clockwise, with North as 0 (zero) degrees and East as 90 degrees. | `int` |
| PRES | Sea level pressure (hPa). | `int` |
| ATMP | Air temperature (Celsius).| `float` |
| WTMP | Sea surface temperature (Celsius). | `float` |
| DEWP | Dewpoint temperature taken at the same height as the air temperature measurement. | `float` |
| VIS | Station visibility (nautical miles). | `float` |
| TID | The water level in feet above or below Mean Lower Low Water (MLLW). | `float` |

## Data Cleaning
### Scores

Scores data were extensively cleaned. Round and Heat values were reformatted for consistency, e.g. "Semifinal" and "Semifinals" were changed to "Semifinals", "Seeding Round" and "Elimination Round" were combined into "Round 1", and other changes described in the code.

Heat Dates were converted to datetime format using `pd.to_datetime()`. To assign a precise time of ocurrence to each heat, the data was revied manually and an offset was applied so that each day's heats would start at 6 am.

Finally scores and datetime data was merged succesfully, grouping by time and gettting the resulting average score per heat, which is our **target**.

### NOAA waves and wind

Waves and wind data were also extensively cleaned. NOAA records missing data as different versions of 99, 99.0, 999, 9999.0, etc. These were removed. In some cases, all data in a column were missing, so these columns were removed. Other missing values such as "..." and "MM" were manually identified using `.unique()` and removed.

Columns were renamed for human readability, and dates were parsed from separate columns into a single column and moved to the index.

Wave and wind directions were converted to `np.sin()` and `np.cos()` for better model usability.

## Feature Engineering
The main feature engineering performed in this project was to create time-shifted features of our buoy data.

There is an inherent delay between measurements at offshore buoys and conditions at the beach, which directly affects our target: **scores at the beach**.

## Findings
There is some predictive power in the models used, as they all perform above baseline. Linear Regression with Lasso performed the best on test data, and produced good insights through feature coefficients that were 'snapped' to 0.

## Risks, Limitations and Assumptions
There are several risks, limitations and assumptions in this project. The final data set used for modeling is relatively small, with only 300+ rows of data. There is about 300 more data points that must be gathered manually and were excluded in this first version of the project. Including this data could improve model performance.

Wave scores are a subjective measure and vary between athletes and judges, one of the main assumptions of the project is that the scores are dependent upon wave conditions, and while this is true, there are other factors that play a part in wave scores, and the scale for scoring is constantly being adjusted by judges depending on the conditions of the day.

Athlete performance can also skew scores. There athletes who can skew the distribution of scores given their dominance of the sport, as can be seen in the EDA section "Scores by Athlete". A good idea for the future is to adjust our target variable 'scores' by each athlete's average score to account for this.

## Notebooks

1. Code:
    1. [WSL Scores Scraper](./code/01-wsl-scraper.ipynb)
    1. [Bouy Data](./code/02-pipeline-buoys.ipynb)
    1. [Wind Data](./code/03-honolulu-wind.ipynb)
    1. [Clean and Merge All Data](./code/04-clean-and-merge-all-data.ipynb)
    1. [Exploratory Data Analysis](./code/05-eda.ipynb)
    1. [Linear Regression Models](./code/06-linear-models.ipynb)
    1. [eXtreme Gradient Boosting Models](./code/07-xgboost-models.ipynb)
    1. [Neural Networks Models](./code/08-neural-networks.ipynb)
    1. [Summary of Model Results](./code/09-models-summary.ipynb)
    1. [Streamlit App DEMO](./code/streamlit_app.py)


2. Data:
    1. [Scraped WSL Data](./data/wsl/avg_scores_all_years.csv)
    1. [Scraped Pipeline Dates Data](./data/wsl/pipe_masters_heat_dates.csv)
    1. [Full Buoy Data](./data/noaa/pipeline_buoy_full.csv)
    1. [Full Wind Data](./data/wind/oahu_wind.csv)
    1. [Cleaned and Merged Data](./data/merged_data.csv)


3. Model Results:
    1. [Linear Models](./model-results/linreg.csv)
    1. [XGBoost Models](./model-results/xgboost.csv)
    1. [Neural Networks Models](./model-results/neuralnetworks.csv)


4. [Presentation](./presentation/capstone-presentation.pdf)


5. [Saved Models](./saved-models/)