# Energy Consumption Prediction System

This project implements an advanced energy consumption prediction system using multiple machine learning and time series models. The system predicts electricity demand based on weather conditions, time factors, and other relevant features.

## Models Implemented

The system includes the following prediction models:

1. **Random Forest** - Ensemble learning method for regression
2. **XGBoost** - Gradient boosting framework
3. **LightGBM** - Gradient boosting framework with tree-based learning
4. **ARIMA** - AutoRegressive Integrated Moving Average for time series forecasting
5. **LSTM** - Long Short-Term Memory neural networks for sequence prediction

## Features

- Hourly energy demand prediction
- Weather-based energy consumption forecasting
- Time-based patterns analysis (hour of day, day of week, seasonality)
- Energy source distribution visualization
- Daily energy balance calculation
- Historical comparison

## Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

## Project Structure

- `app.py` - Main Streamlit application
- `model_development.py` - Model training and evaluation
- `data_preprocessing.py` - Data cleaning and feature engineering
- `get_data.py` - Data loading functions
- `notebooks/` - Jupyter notebooks for analysis
- `datasets/` - Input data files
- `artifacts/` - Saved models and scalers

## Time Series Models

### ARIMA Model

The ARIMA (AutoRegressive Integrated Moving Average) model is used for time series forecasting. It combines:
- AR (AutoRegressive) component - uses the relationship between an observation and a number of lagged observations
- I (Integrated) component - differencing to make the time series stationary
- MA (Moving Average) component - uses the relationship between an observation and a residual error from a moving average model

### LSTM Model

The LSTM (Long Short-Term Memory) neural network is designed to recognize patterns in sequence data. It's particularly effective for:
- Capturing long-term dependencies in time series
- Learning complex patterns in energy consumption
- Handling seasonal and cyclical patterns

## Data Preprocessing for Time Series

The project includes specialized preprocessing for time series data:
- Stationarity testing and transformation
- Time series decomposition (trend, seasonality, residual)
- Sequence creation for LSTM
- Feature scaling for neural networks

## Model Performance

The system evaluates models using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Contributors

- [Your Name]

## License

This project is licensed under the MIT License - see the LICENSE file for details. 