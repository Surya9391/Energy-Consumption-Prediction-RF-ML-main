import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

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

def prepare_time_series_data(data, target_col='Electricity Demand (MW)', date_col='Date'):
    """
    Prepare data for time series analysis (ARIMA and LSTM)
    
    Parameters:
    -----------
    data : pandas DataFrame
        The input data containing time series
    target_col : str
        The name of the target column
    date_col : str
        The name of the date column
        
    Returns:
    --------
    ts_data : pandas DataFrame
        Data prepared for time series analysis
    """
    # Ensure data is sorted by date
    ts_data = data.copy()
    if date_col in ts_data.columns:
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        ts_data = ts_data.sort_values(date_col)
        
        # Set date as index for time series analysis
        ts_data = ts_data.set_index(date_col)
    
    return ts_data

def check_stationarity(time_series):
    """
    Simplified check for stationarity by comparing mean and variance across segments
    
    Parameters:
    -----------
    time_series : pandas Series
        The time series to check
        
    Returns:
    --------
    is_stationary : bool
        True if the time series appears stationary, False otherwise
    p_value : float
        A placeholder p-value (not actually calculated)
    """
    # Split the series into segments
    segments = 3
    segment_size = len(time_series) // segments
    
    means = []
    variances = []
    
    # Calculate mean and variance for each segment
    for i in range(segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < segments - 1 else len(time_series)
        segment = time_series[start:end]
        means.append(segment.mean())
        variances.append(segment.var())
    
    # Check if means and variances are similar across segments
    mean_diff = max(means) - min(means)
    var_diff = max(variances) - min(variances)
    
    # Simplified criteria for stationarity
    is_stationary = (mean_diff / abs(np.mean(means)) < 0.1) and (var_diff / abs(np.mean(variances)) < 0.1)
    
    return is_stationary, 0.05  # Placeholder p-value

def make_stationary(time_series, max_diff=2):
    """
    Make a time series stationary by differencing
    
    Parameters:
    -----------
    time_series : pandas Series
        The time series to make stationary
    max_diff : int
        Maximum number of differencing operations
        
    Returns:
    --------
    stationary_series : pandas Series
        The stationary time series
    d : int
        The number of differencing operations performed
    """
    series = time_series.copy()
    d = 0
    
    # Check initial stationarity
    is_stationary, _ = check_stationarity(series)
    
    # Apply differencing until stationary or max_diff reached
    while not is_stationary and d < max_diff:
        series = series.diff().dropna()
        d += 1
        is_stationary, _ = check_stationarity(series)
    
    return series, d

def decompose_time_series(time_series, period=24):
    """
    Simplified decomposition of time series into trend and seasonal components
    
    Parameters:
    -----------
    time_series : pandas Series
        The time series to decompose
    period : int
        The seasonal period (e.g., 24 for hourly data with daily seasonality)
        
    Returns:
    --------
    decomposition : dict
        Dictionary containing trend, seasonal, and residual components
    """
    # Ensure no missing values
    time_series = time_series.fillna(method='ffill')
    
    # Calculate trend using rolling mean
    trend = time_series.rolling(window=period, center=True).mean()
    
    # Fill NaN values at the beginning and end
    trend = trend.fillna(method='bfill').fillna(method='ffill')
    
    # Calculate seasonal component
    # For each position in the period, calculate the average deviation from trend
    seasonal = pd.Series(index=time_series.index)
    
    for i in range(period):
        seasonal_indices = [x for x in range(len(time_series)) if x % period == i]
        seasonal_values = time_series.iloc[seasonal_indices] - trend.iloc[seasonal_indices]
        seasonal_mean = seasonal_values.mean()
        
        for idx in seasonal_indices:
            if idx < len(seasonal):
                seasonal.iloc[idx] = seasonal_mean
    
    # Calculate residual
    residual = time_series - trend - seasonal
    
    return {
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual
    }

def prepare_lstm_data(data, target_col, feature_cols, sequence_length=24):
    """
    Prepare data for LSTM model by creating sequences
    
    Parameters:
    -----------
    data : pandas DataFrame
        The input data
    target_col : str
        The name of the target column
    feature_cols : list
        List of feature column names
    sequence_length : int
        The length of sequences to create
        
    Returns:
    --------
    X : numpy array
        The input sequences for LSTM
    y : numpy array
        The target values
    """
    # Extract features and target
    X_data = data[feature_cols].values
    y_data = data[target_col].values
    
    # Create sequences
    X, y = [], []
    for i in range(len(X_data) - sequence_length):
        X.append(X_data[i:i+sequence_length])
        y.append(y_data[i+sequence_length])
    
    return np.array(X), np.array(y)

def scale_time_series_data(X_train, X_test, y_train, y_test):
    """
    Scale features and target for time series models
    
    Parameters:
    -----------
    X_train, X_test : numpy arrays
        The training and test features
    y_train, y_test : numpy arrays
        The training and test targets
        
    Returns:
    --------
    X_train_scaled, X_test_scaled : numpy arrays
        The scaled features
    y_train_scaled, y_test_scaled : numpy arrays
        The scaled targets
    feature_scaler, target_scaler : StandardScaler
        The fitted scalers
    """
    # Scale features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale target
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler
