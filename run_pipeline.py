import pandas as pd
import numpy as np
from src.get_data import get_data 
from src.data_preprocessing import data_cleaning
from src.model_dovelopment import model_dovelopment


def model_pipeline():
    energy_data = pd.read_csv('D:\FREELANCE_PROJECTS\energy-consumption-forecast-lstm\datasets,\energy_dataset.csv', encoding='UTF-8-SIG')
    weather_data = pd.read_csv('D:\FREELANCE_PROJECTS\energy-consumption-forecast-lstm\datasets,\weather_features.csv', encoding='UTF-8-SIG')
    return energy_data, weather_data


def process_post():
    energy_data, weather_data = model_pipeline()
    print("Pipeline run successfully!")
    data1, data2 = get_data(energy_data, weather_data)
    print(data1, data2)
    print("Get Data successfully!")
    cleaned_data = data_cleaning(energy_data, weather_data)
    print(cleaned_data)
    print("Data Cleaning successfully!")
    model_dovelopment(cleaned_data)

if __name__ == "__main__":
    process_post()


    

