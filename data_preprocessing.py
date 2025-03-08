import pandas as pd
import numpy as np
from datetime import datetime

def is_festive_season(date):
    """
    Check if the date falls in major festival seasons in Andhra Pradesh
    """
    date = pd.to_datetime(date)
    month = date.month
    day = date.day
    
    # Major festivals in AP
    festivals = {
        # Sankranti season (January)
        'sankranti': (month == 1) and (13 <= day <= 15),
        # Ugadi season (March-April)
        'ugadi': (month == 3 and day >= 20) or (month == 4 and day <= 5),
        # Dussehra season (September-October)
        'dussehra': (month == 9 and day >= 25) or (month == 10 and day <= 5),
        # Diwali season (October-November)
        'diwali': (month == 10 and day >= 20) or (month == 11 and day <= 5),
    }
    
    return any(festivals.values())

def calculate_regional_heat_index(temperature, humidity, city):
    """
    Calculate heat index with regional factors for coastal AP cities
    """
    # Base heat index calculation
    temperature_f = temperature * 9/5 + 32
    hi = 0.5 * (temperature_f + 61.0 + ((temperature_f - 68.0) * 1.2) + (humidity * 0.094))
    hi_celsius = (hi - 32) * 5/9
    
    # City-specific coastal and industrial factors
    city_factors = {
        'Kakinada': 1.2,      # Major port city, industrial area
        'Surampalem': 1.1,    # Industrial area
        'Rajahmundry': 1.15,  # Inland city, higher temperatures
        'Samarlakota': 1.1    # Industrial area
    }
    
    return hi_celsius * city_factors.get(city, 1.0)

def calculate_energy_mix_factor(hour, solar_radiation):
    """
    Calculate energy mix factor based on time of day and solar radiation
    """
    # Base factor
    mix_factor = 1.0
    
    # Time of day effect on renewable efficiency
    is_daytime = 6 <= hour <= 18
    
    # Solar effectiveness during daytime
    if is_daytime:
        solar_effectiveness = solar_radiation / 1000  # Normalize solar radiation
        mix_factor *= (1 - (solar_effectiveness * 0.3))
            
    return mix_factor

def data_cleaning(climate_data, monthly_data):
    # Merge climate data with monthly data on Date
    merged_data = pd.merge(climate_data, monthly_data, on='Date', how='left')
    
    # Filter for relevant cities only
    valid_cities = ['Kakinada', 'Surampalem', 'Rajahmundry', 'Samarlakota']
    merged_data = merged_data[merged_data['City'].isin(valid_cities)]
    
    # Ensure city names are standardized
    merged_data['City'] = merged_data['City'].str.strip()
    merged_data['City'] = merged_data['City'].replace({
        'KAKINADA': 'Kakinada',
        'SURAMPALEM': 'Surampalem',
        'RAJAHMUNDRY': 'Rajahmundry',
        'SAMARLAKOTA': 'Samarlakota',
        'Samalkot': 'Samarlakota',
        'SAMALKOT': 'Samarlakota'
    })
    
    # Add seasonal patterns specific to coastal AP
    merged_data['Season'] = pd.to_datetime(merged_data['Date']).dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Spring',
        3: 'Spring', 4: 'Summer', 5: 'Summer',
        6: 'Summer', 7: 'Monsoon', 8: 'Monsoon',
        9: 'Monsoon', 10: 'Post_Monsoon', 11: 'Post_Monsoon'
    })
    
    # Extract hour from Date
    merged_data['Hour'] = pd.to_datetime(merged_data['Date']).dt.hour
    
    # Add festive season indicator
    merged_data['Is_Festival'] = merged_data['Date'].apply(is_festive_season)
    
    # Add peak load patterns (adjusted for local patterns)
    merged_data['Peak_Type'] = merged_data['Hour'].map(lambda x: 
        'Super_Peak' if x in [10, 11, 12, 13, 14, 15] else  # Adjusted for local peak hours
        'Peak' if x in [7, 8, 9, 16, 17, 18, 19] else
        'Off_Peak'
    )
    
    # Select relevant features
    selected_features = [
        'City',
        'Temperature (°C)',
        'Humidity (%)',
        'Wind Speed (km/h)',
        'Rainfall (mm)',
        'Solar Radiation (W/m²)',
        'Sector',
        'Renewable_Share_%',    # Total renewable percentage
        'Thermal_Share_%',      # Thermal percentage
        'Grid_Frequency_Hz',
        'Hour',
        'Season',
        'Peak_Type',
        'Is_Festival'
    ]
    
    # Add target variable
    target = ['Electricity Demand (MW)']
    
    # Create final dataset
    final_data = merged_data[selected_features + target].copy()
    
    # Add energy mix factor
    final_data['Energy_Mix_Factor'] = final_data.apply(
        lambda row: calculate_energy_mix_factor(
            row['Hour'],
            row['Solar Radiation (W/m²)']
        ), axis=1
    )
    
    # Add interaction features
    final_data['Temperature_Humidity'] = final_data['Temperature (°C)'] * final_data['Humidity (%)'] / 100
    final_data['Regional_Heat_Index'] = final_data.apply(
        lambda row: calculate_regional_heat_index(
            row['Temperature (°C)'], 
            row['Humidity (%)'], 
            row['City']
        ), axis=1
    )
    
    # Add temperature threshold effects (demand increases more rapidly above certain temperatures)
    final_data['High_Temp_Effect'] = (final_data['Temperature (°C)'] > 35).astype(int) * \
                                    (final_data['Temperature (°C)'] - 35) * 1.5
    
    # Solar and wind features
    final_data['Solar_Efficiency'] = final_data['Solar Radiation (W/m²)'] * \
                                    (1 - final_data['Humidity (%)']/100) * \
                                    (1 - final_data['Rainfall (mm)'].clip(0, 25)/25)
    final_data['Wind_Power_Potential'] = final_data['Wind Speed (km/h)']**3
    
    # Add time-based features
    final_data['Month'] = pd.to_datetime(merged_data['Date']).dt.month
    final_data['Is_Summer'] = final_data['Season'] == 'Summer'
    final_data['Is_Peak_Hour'] = final_data['Peak_Type'].isin(['Super_Peak', 'Peak'])
    
    # Add city-specific industrial load factors
    city_industrial_factors = {
        'Kakinada': 1.25,     # Major industrial port city
        'Surampalem': 1.15,   # Educational hub
        'Rajahmundry': 1.1,   # Commercial center
        'Samarlakota': 1.2    # Industrial area
    }
    final_data['Industrial_Load_Factor'] = final_data['City'].map(city_industrial_factors)
    
    # Convert categorical variables to dummy variables
    final_data = pd.get_dummies(final_data, columns=['City', 'Sector', 'Season', 'Peak_Type'])
    
    # Fill missing values with more appropriate methods
    numerical_columns = final_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        if col in ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)']:
            # Use rolling mean for weather data
            final_data[col] = final_data[col].fillna(final_data[col].rolling(24, min_periods=1).mean())
        else:
            # Use normal mean for other numerical columns
            final_data[col] = final_data[col].fillna(final_data[col].mean())
    
    return final_data

def calculate_heat_index(temperature, humidity):
    """
    Calculate the heat index (feels-like temperature) using temperature and humidity
    Temperature should be in Celsius, humidity in percentage
    """
    # Convert Celsius to Fahrenheit for the standard heat index formula
    temperature_f = temperature * 9/5 + 32
    
    # Simple heat index formula
    hi = 0.5 * (temperature_f + 61.0 + ((temperature_f - 68.0) * 1.2) + (humidity * 0.094))
    
    # Convert back to Celsius
    return (hi - 32) * 5/9
