

from flask import Flask, request, jsonify
import openmeteo_requests
import numpy as np
import requests_cache
import pandas as pd
from retry_requests import retry
import requests
# Import the Python SDK
import google.generativeai as genai
# Used to securely store your API key
from google.colab import userdata
#import typing_extensions as typing
from typing_extensions import TypedDict  # Import from typing_extensions
import json
import calendar

import math
from datetime import datetime





app = Flask(__name__)

def calculateGDD(T_max, T_min, T_base):
    return abs((T_max+T_min)/2 - T_base)

def calculatePET(T, humidity):
    exponent = (17.27 * T) / (T + 237.3)
    e_s = 0.6108 * np.exp(exponent)
    e_a = (e_s * humidity) / 100
    delta = (4098*e_s)/(T+237.3)**2
    c_p = 1010  #Specific heat of air = 1010 J/kg.K
    P = 101.325  #Atmospheric pressure = 101.325 hPa
    l_v = 2.45 * 10**6  #Latent heat of vaporization = 2.45*10^6 J/kg
    psy_con = (c_p*P)/(0.622*l_v)  #Psychrometric constant = 0.622
    r_n = 0
    G = 0

    avg_wind_speed = 2 # hourly_dataframe['wind_speed_10m'].mean()

    return (0.408 * delta * (r_n - G) + psy_con * (900 / (T + 273)) * avg_wind_speed * (e_s - e_a)/(delta+psy_con * (1+0.34*avg_wind_speed)))

def calculateSoilMoisture(Precipitation, man_watering, month_days, area,  PET):
    return abs(Precipitation + (man_watering/(month_days*area)) - PET)

def calculateTemperatureStressIndex(T_max, T_opt, T_thres):
    return (T_max - T_opt)/(T_thres - T_opt)
def calculateFrostRisk(T_thres, T_min):
    return max(0, T_thres-T_min)
def calculatePhotosyntehis(LUE, sun_hours, growth_duration):
    return LUE * sun_hours * growth_duration
def calculateGrowthIndex(GDD, SoilMoisture, Max_Soil_Moisture, Photosynthesis, Max_Photosynthesis):
    return (GDD * (SoilMoisture/Max_Soil_Moisture) * (Photosynthesis/Max_Photosynthesis))


def calGrowthIndex():
    cache_session.cache.clear()
    data_growth_index = request.json
    lat = float(data_growth_index.get('latitude', 0))
    lon = float(data_growth_index.get('longitude', 0))
    tree_name = str(data_growth_index.get('tree_name', 0))
    area = float(data_growth_index.get('area_of_plantation', 0))
    man_watering = float(data_growth_index.get('manual_watering', 0))
    start = str(data_growth_index.get('start_date', 0))

    today_date = datetime.today().date()

    print(today_date)
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(start),
        "end_date": str(today_date),
        "hourly": ["temperature_2m", "relative_humidity_2m","precipitation", "wind_speed_10m"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "daylight_duration"],
        "timezone": "GMT"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    hourly_dataframe.fillna(0, inplace=True)
    hourly_dataframe.interpolate(method='linear', inplace=True)
    print(hourly_dataframe)

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_daylight_duration = daily.Variables(2).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["daylight_duration"] = daily_daylight_duration

    daily_dataframe = pd.DataFrame(data = daily_data)
    daily_dataframe.fillna(0, inplace=True)
    daily_dataframe.interpolate(method='linear', inplace=True)
    print(daily_dataframe)

    
    baseURL = "https://api.bigdatacloud.net/data/reverse-geocode-client?"
    endURL = "&localityLanguage=en"

    url = f"{baseURL}latitude={str(lat)}&longitude={str(lon)}{endURL}"

    # url = f"{baseURL}latitude=40&longitude=-70{endURL}"
    # print(url)

    city_name = requests.get(url).json()['localityInfo']['administrative'][3]["name"]
    print(city_name)
    city_description = requests.get(url).json()['localityInfo']['administrative'][3]["description"]
    print(city_description)

    class Tree(TypedDict):  # Updated Tree class
        tree_name: str
        optimal_temperature: float  # Add a field for the optimal
        temperature_stress: float  # Add a field for the temperature stress index
        frost_risk: float  # Add a field for the frost risk
        LUE_value: float  # Add a field for the LUE value
        growth_duration: float  # Add a field for the growth duration


    # Assuming the user data is available in the context
    GOOGLE_API_KEY = 'AIzaSyAlGbR6CHZNxCyx2MOCc74GV1GuGppvGN0'   #userdata.get('GOOGLE_API_KEY')

    # Configure the GenAI model with the API key
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro', generation_config={"response_mime_type": "application/json", "response_schema": list[Tree]})

    # Define the variables for tree name and city details
    tree_name = "Oak"  # Example tree name
    city_name = "San Francisco"  # Example city name
    city_description = "a coastal city with a mild climate"

    # Generate the prompt using the defined variables
    prompt = f"""
    We are planting a {tree_name} tree in {city_name}, {city_description}. Give me an accurate average value of:
    Optimal temperature, Temperature beyond which stress occurs, Frost risk, Light Use Efficiency and growth duration of {tree_name}.
    """

    print("The prompt generated is:" + prompt)

    # Generate the response from the model
    response = model.generate_content(prompt)

    # Assuming the response is already in JSON format
    response_data = json.loads(response.text)  # Use the JSON parser to handle the structured response

    # Extract the temperature value from the response
    output_value_list = []
    for val in response_data:
        output_value_list.append(val["optimal_temperature"])  # Adjust this key to match the actual response schema
        output_value_list.append(val["temperature_stress"])  # Adjust this key to match the actual response schema
        output_value_list.append(val["frost_risk"])  # Adjust this key to match the actual response schema
        output_value_list.append(val["LUE_value"])  # Adjust this key to match the actual response schema
        output_value_list.append(val["growth_duration"])  # Adjust this key to match the actual response schema

    # Print the final answer
    print("Answer: ")
    print(f"Optimal temperatures for {tree_name}'s growth: {output_value_list}")


    hourly_wind_speed_10m_array = np.array(hourly_dataframe["wind_speed_10m"])
    daily_temperature_2m_max_array = np.array(daily_dataframe["temperature_2m_max"])
    daily_temperature_2m_min_array = np.array(daily_dataframe["temperature_2m_min"])
    hourly_temperature_2m_array = np.array(hourly_dataframe["temperature_2m"])
    hourly_relative_humidity_2m_array = np.array(hourly_dataframe["relative_humidity_2m"])
    hourly_precipitation_array = np.array(hourly_dataframe["precipitation"])
    daily_daylight_duration_array = np.array(daily_dataframe["daylight_duration"])

    T_max = daily_temperature_2m_max_array.mean()
    print(f"T_max = {T_max}")
    T_min = daily_temperature_2m_min_array.mean()
    print(f"T_min = {T_min}")
    T_base = 0
    T=(T_max+T_min)/2
    H=hourly_relative_humidity_2m_array.mean()
    print(f"H = {H}")
    Precipitation = hourly_precipitation_array.mean()
    print(f"Precipitation = {Precipitation}")
    sun_hours = ((daily_daylight_duration_array.mean())/60)/60
    print(f"sun_hours = {sun_hours}")


    now = datetime.now()
    month = now.month
    year = now.year

    month_days = calendar.monthrange(year, month)[1]


    PET = 0



    T_opt = output_value_list[0]
    T_thres_stress = output_value_list[1]
    T_thres_frost = output_value_list[2]


    LUE = output_value_list[3]
    Growth_duration = (output_value_list[4])/365



    GDD = calculateGDD(T_max, T_min, T_base)
    print(f"GDD = {GDD}")
    PET = calculatePET(T, H)
    print(f"PET = {PET}")
    SoilMoisture = calculateSoilMoisture(Precipitation, man_watering, month_days, area, PET)
    print(f"SoilMoisture = {SoilMoisture}")
    Max_Soil_Moisture = 20 #max(SoilMoisture)
    TemperatueStressIndex = calculateTemperatureStressIndex(T_max, T_opt, T_thres_stress)
    FrostRisk = calculateFrostRisk(T_thres_frost, T_min)
    Photosynthesis = calculatePhotosyntehis(LUE, sun_hours, Growth_duration)
    Max_Photosynthesis = 10 # max(SoilMoisture)
    print(f"Photosynthesis = {Photosynthesis}")


    Growth_index = calculateGrowthIndex(GDD, SoilMoisture, Max_Soil_Moisture, Photosynthesis, Max_Photosynthesis)

    print(Growth_index)
    result_growth = {
        'latitude': str(lat),
        'longitude': str(lon),
        'tree_name': str(tree_name),
        'area_of_plantation': str(area),
        'manual_watering': str(man_watering),
        'start_date': str(start),
        'growth_index': str(Growth_index)
    }
    return (result_growth)



@app.route('/run-script', methods=['POST'])
def run_script():
    
    result_json = calculateGrowthIndex()
    
    return jsonify(result_json)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
