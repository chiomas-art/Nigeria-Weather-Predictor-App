# 🌦️ Nigeria Weather Predictor

A machine learning web app that predicts the **typical weather** (temperature, humidity, air pressure and wind speed) for any of Nigeria's 36 states and the FCT, on any chosen date and time of day.

🔗 **Live app:** [paste your Streamlit link here]

## What it does

- Pick a **state**, a **date** and a **time of day** (Morning, Afternoon or Evening)
- Click **Predict weather** to get temperature (°C), humidity (%), pressure (hPa) and wind speed (m/s)
- See the whole day (morning, afternoon, evening) in one table
- Check the **Model accuracy** tab for the test scores
- Check the **Data** tab for the dataset summary and a monthly temperature chart for each state

> **Note:** The app predicts the *typical* weather for a place at a time of year, learned from past years. It is not an exact forecast for a specific day.

## Data

- **Source:** [Open-Meteo](https://open-meteo.com) Historical Weather API (free, no API key)
- **Period:** 1 January 2021 to 31 December 2025
- **Locations:** the capital city of each of the 36 states, plus Abuja (FCT)
- **Readings used:** 6:00 (Morning), 15:00 (Afternoon) and 18:00 (Evening)
- **Size:** 202,686 readings
- The app downloads the data automatically the first time it runs and saves it as `nigeria_weather_2021_2025.csv`

## Model

- **Algorithm:** Random Forest Regression (scikit-learn), predicting all four weather values at once
- **Inputs:** latitude, longitude, hour of the day, and day of the year (as sine and cosine so the seasons repeat in a cycle)
- **Training and testing:** trained on 2021 to 2024 and tested on 2025, a year the model had never seen. The final model is then retrained on all five years.

### Results (2025 test year)

| Predicted value | R² score | Average error |
|---|---|---|
| Temperature (°C) | 0.86 | 1.45 °C |
| Humidity (%) | 0.88 | 7.17 % |
| Pressure (hPa) | 0.645 | 1.25 hPa |
| Wind speed (m/s) | 0.246 | 0.77 m/s |

Temperature and humidity are predicted well. Wind speed is the weakest, because wind changes randomly from day to day.

## Tech stack

Python · Streamlit · pandas · NumPy · scikit-learn · requests

## Run it locally

```bash
git clone <your-repository-link>
cd <repository-folder>
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the weather data for all 37 locations, which takes a few minutes. Later runs start quickly.

## Project files

| File | Purpose |
|---|---|
| `app.py` | The complete app: data download, model training and the Streamlit interface |
| `requirements.txt` | The Python libraries the app needs |

## Limitations

- Gives typical weather, not an exact forecast
- Each state is represented by its capital city only
- Wind speed predictions are rough
- Only covers 2021 to 2025 and three times of day

## Possible improvements

- More years of data and hourly predictions
- Add rainfall
- Compare other models, such as Gradient Boosting
- Show a real forecast beside the typical value

## Author

Proudly Chiomas-Art, check out the App using the [ https://nigeria-weather-predictor-app-fbflbk94azr6o8l6szjplw.streamlit.app/ ]

Thank Youu.
