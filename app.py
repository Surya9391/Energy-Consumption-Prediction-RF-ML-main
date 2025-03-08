import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from get_data import get_data
from data_preprocessing import data_cleaning, calculate_heat_index, calculate_regional_heat_index
from model_development import model_development, plot_model_comparison, plot_feature_importance
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Energy Demand Prediction", layout="wide")
st.title('Forecasting Energy Demand Prediction')

# Load and process data
@st.cache_data
def load_data():
    climate_data_path = "new datasets/Final_Energy_Prediction_Climate_Data_Corrected.csv"
    monthly_data_dir = "new datasets"
    climate_data, monthly_data = get_data(climate_data_path, monthly_data_dir)
    final_data = data_cleaning(climate_data, monthly_data)
    
    # Get the actual city columns from the processed data
    city_columns = [col for col in final_data.columns if col.startswith('City_')]
    cities = [col.replace('City_', '') for col in city_columns]
    
    return final_data, cities

# Load data and train models
if 'models' not in st.session_state:
    with st.spinner('Loading data and training models...'):
        final_data, cities = load_data()
        models, scaler, model_scores, feature_importance = model_development(final_data)
        
        st.session_state.models = models
        st.session_state.scaler = scaler
        st.session_state.scores = model_scores
        st.session_state.feature_importance = feature_importance
        st.session_state.cities = cities
        st.session_state.training_data = final_data
        st.session_state.feature_names = final_data.drop('Electricity Demand (MW)', axis=1).columns.tolist()
        # Store the city columns separately
        st.session_state.city_columns = [col for col in st.session_state.feature_names if col.startswith('City_')]

# Model Performance Section
st.header('Model Performance Comparison')
col1, col2 = st.columns([2, 1])

with col2:
    # Model selection
    selected_model = st.selectbox(
        'Select Model for Prediction',
        ['Random Forest', 'XGBoost', 'LightGBM'],
        help="Choose the model to use for prediction"
    )

    # Show selected model's performance
    model_metrics = st.session_state.scores[selected_model]
    st.markdown(f"""
    ### Selected Model Performance
    <div style='padding: 15px; border-radius: 10px; background-color: #f0f2f6;'>
        <h4>Accuracy Metrics:</h4>
        <ul>
            <li><b>R² Score:</b> {model_metrics['R2']*100:.1f}%</li>
            <li><b>RMSE:</b> {model_metrics['RMSE']:.2f} MW</li>
            <li><b>MAE:</b> {model_metrics['MAE']:.2f} MW</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Model information
    model_info = {
        'Random Forest': """
        - Ensemble learning method
        - Good for handling non-linear relationships
        - Feature importance capabilities
        """,
        'XGBoost': """
        - Gradient boosting implementation
        - High performance and speed
        - Advanced regularization
        """,
        'LightGBM': """
        - Light Gradient Boosting Machine
        - Faster training speed
        - Better accuracy
        """
    }
    
    st.markdown(f"""
    ### Model Information
    {model_info[selected_model]}
    """)

with col1:
    # Plot model comparison
    comparison_fig = plot_model_comparison(st.session_state.scores)
    st.plotly_chart(comparison_fig)

    # Show feature importance for tree-based models
    if selected_model in ['Random Forest', 'XGBoost', 'LightGBM']:
        st.subheader(f'Feature Importance - {selected_model}')
        feature_importance_fig = plot_feature_importance(
            st.session_state.feature_importance[selected_model],
            selected_model
        )
        st.plotly_chart(feature_importance_fig)

# Sidebar inputs
st.sidebar.header('Enter Input Parameters')

# Location selection
selected_city = st.sidebar.selectbox('Select City', ['Kakinada', 'Surampalem', 'Rajahmundry', 'Samarlakota'])

# Time-based inputs
st.sidebar.subheader('Date and Time Selection')

# Date selection with extended range
max_prediction_date = datetime(2027, 12, 31).date()
selected_date = st.sidebar.date_input(
    "Select Date",
    value=datetime.now().date(),
    min_value=datetime.now().date(),
    max_value=max_prediction_date
)

# Add prediction confidence indicator
days_ahead = (selected_date - datetime.now().date()).days
confidence_level = "High" if days_ahead <= 7 else "Medium" if days_ahead <= 30 else "Low"
confidence_color = "#00FF00" if confidence_level == "High" else "#FFA500" if confidence_level == "Medium" else "#FF0000"

st.sidebar.markdown(f"""
<div style='padding: 10px; border-radius: 5px; background-color: {confidence_color}30;'>
    <p><b>Prediction Confidence:</b> {confidence_level}</p>
    <p><small>Based on {days_ahead} days ahead of current date</small></p>
</div>
""", unsafe_allow_html=True)

# Time selection
selected_time = st.sidebar.time_input(
    "Select Time",
    value=datetime.now().time(),
)

# Climate parameters
st.sidebar.subheader('Climate Parameters')
temperature = st.sidebar.slider('Temperature (°C)', 0.0, 50.0, 25.0)
humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 60.0)
wind_speed = st.sidebar.slider('Wind Speed (km/h)', 0.0, 50.0, 15.0)
rainfall = st.sidebar.slider('Rainfall (mm)', 0.0, 50.0, 5.0)

# Energy Source Parameters
st.sidebar.subheader('Energy Source Parameters')
col1, col2 = st.sidebar.columns(2)

with col1:
    st.markdown("##### Renewable Sources")
    solar_power = st.number_input('Solar Power (MW)', 0.0, 1000.0, 100.0)
    wind_power = st.number_input('Wind Power (MW)', 0.0, 1000.0, 50.0)
    hydro_power = st.number_input('Hydro Power (MW)', 0.0, 1000.0, 200.0)

with col2:
    st.markdown("##### Conventional Sources")
    thermal_power = st.number_input('Thermal Power (MW)', 0.0, 2000.0, 500.0)
    grid_power = st.number_input('Grid Power (MW)', 0.0, 2000.0, 400.0)

# Calculate energy mix
total_power = solar_power + wind_power + hydro_power + thermal_power + grid_power
renewable_share = ((solar_power + wind_power + hydro_power) / total_power * 100) if total_power > 0 else 0
thermal_share = (thermal_power / total_power * 100) if total_power > 0 else 0
grid_share = (grid_power / total_power * 100) if total_power > 0 else 0

if st.sidebar.button('Predict Energy Demand'):
    # Prepare input data
    hour = selected_time.hour
    month = selected_date.month
    
    # Initialize input data with all features including all city columns as 0
    input_data = pd.DataFrame(0, index=[0], columns=st.session_state.feature_names)
    
    # Set features
    regional_heat_index = calculate_regional_heat_index(temperature, humidity, selected_city)
    temperature_humidity = temperature * humidity / 100
    solar_efficiency = 100 * (1 - humidity/100) * (1 - rainfall/25 if rainfall <= 25 else 0)
    wind_power_potential = wind_speed**3
    is_summer = 1 if month in [4, 5, 6, 7, 8, 9] else 0
    is_peak_hour = 1 if hour in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] else 0
    
    # Set all non-city features
    input_data['Temperature (°C)'] = temperature
    input_data['Humidity (%)'] = humidity
    input_data['Wind Speed (km/h)'] = wind_speed
    input_data['Rainfall (mm)'] = rainfall
    input_data['Renewable_Share_%'] = renewable_share
    input_data['Thermal_Share_%'] = thermal_share
    input_data['Temperature_Humidity'] = temperature_humidity
    input_data['Regional_Heat_Index'] = regional_heat_index
    input_data['Solar_Efficiency'] = solar_efficiency
    input_data['Wind_Power_Potential'] = wind_power_potential
    input_data['Hour'] = hour
    input_data['Month'] = month
    input_data['Is_Summer'] = is_summer
    input_data['Is_Peak_Hour'] = is_peak_hour

    # Set the selected city to 1, ensure we use only cities from training data
    city_column = f'City_{selected_city}'
    if city_column in st.session_state.city_columns:
        input_data[city_column] = 1
    else:
        st.error(f"Error: City '{selected_city}' was not present in the training data. Please select a different city.")
        st.stop()
    
    # Scale input data
    input_scaled = st.session_state.scaler.transform(input_data)
    
    # Make prediction
    prediction = st.session_state.models[selected_model].predict(input_scaled)[0]
    
    # Display results in a container with better styling
    st.markdown("---")
    st.header("Prediction Results")
    
    # Main prediction container
    with st.container():
        # Determine demand level and color
        demand_color = (
            "#FF0000" if prediction > total_power else
            "#FFA500" if prediction > 0.8 * total_power else
            "#FFFF00" if prediction > 0.6 * total_power else
            "#00FF00"
        )
        
        demand_level = (
            "CRITICAL" if prediction > total_power else
            "HIGH" if prediction > 0.8 * total_power else
            "MODERATE" if prediction > 0.6 * total_power else
            "LOW"
        )
        
        # Create two rows for better organization
        row1_col1, row1_col2 = st.columns([1, 2])
        
        with row1_col1:
            st.markdown(
                f"""
                <div style="padding: 25px; border-radius: 15px; background-color: {demand_color}30; margin-bottom: 20px;">
                    <h2 style="color: {demand_color}; margin-bottom: 15px;">Demand Level: {demand_level}</h2>
                    <h3 style="margin-bottom: 10px;">Predicted Energy Demand</h3>
                    <h2 style="color: #1f77b4;">{prediction:.2f} MW</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Add historical comparison in a styled container
            similar_conditions = st.session_state.training_data[
                (st.session_state.training_data['Temperature (°C)'].between(temperature-5, temperature+5)) &
                (st.session_state.training_data['Humidity (%)'].between(humidity-10, humidity+10)) &
                (st.session_state.training_data[f'City_{selected_city}'] == 1)
            ]
            
            if not similar_conditions.empty:
                avg_demand = similar_conditions['Electricity Demand (MW)'].mean()
                st.markdown(
                    f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin-top: 20px;">
                        <h4 style="color: #1f77b4;">Historical Average</h4>
                        <h3>{avg_demand:.2f} MW</h3>
                        <p style="color: #666; font-size: 0.9em;">Under similar conditions</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        with row1_col2:
            # Energy mix visualization with better styling
            st.markdown('<div style="padding: 15px; border-radius: 10px; background-color: #f8f9fa;">', unsafe_allow_html=True)
            energy_mix = pd.DataFrame({
                'Source': ['Solar', 'Wind', 'Hydro', 'Thermal', 'Grid'],
                'Power (MW)': [solar_power, wind_power, hydro_power, thermal_power, grid_power]
            })
            
            fig = px.pie(energy_mix, values='Power (MW)', names='Source',
                        title='Energy Source Distribution',
                        color='Source',
                        color_discrete_map={
                            'Solar': '#FFD700',
                            'Wind': '#87CEEB',
                            'Hydro': '#4169E1',
                            'Thermal': '#8B4513',
                            'Grid': '#808080'
                        })
            fig.update_layout(
                title_x=0.5,
                title_font_size=20,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(t=60, b=60, l=40, r=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Second row for additional visualizations
        if not similar_conditions.empty:
            st.markdown("### Historical Analysis", unsafe_allow_html=True)
            
            # Show demand distribution
            st.markdown('<div style="padding: 15px; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 20px;">', unsafe_allow_html=True)
            fig = px.histogram(similar_conditions, x='Electricity Demand (MW)',
                             title='Demand Distribution (Similar Conditions)',
                             color_discrete_sequence=['#4169E1'])
            fig.update_layout(
                title_x=0.5,
                title_font_size=16,
                margin=dict(t=40, b=40, l=40, r=40),
                height=400  # Fixed height for better visualization
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Show seasonal patterns
            st.markdown('<div style="padding: 15px; border-radius: 10px; background-color: #f8f9fa;">', unsafe_allow_html=True)
            seasonal_data = st.session_state.training_data.groupby('Month')['Electricity Demand (MW)'].mean()
            fig = px.line(seasonal_data, 
                         title='Monthly Demand Pattern',
                         labels={'value': 'Average Demand (MW)', 'Month': 'Month'},
                         color_discrete_sequence=['#1f77b4'])
            fig.update_layout(
                title_x=0.5,
                title_font_size=16,
                margin=dict(t=40, b=40, l=40, r=40),
                height=400  # Fixed height for better visualization
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Long-term analysis section with better styling
        if days_ahead > 7:
            st.markdown("---")
            st.subheader('Long-term Prediction Analysis')
            
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin-top: 20px;">
                    <h4 style="color: #1f77b4;">Prediction Confidence Analysis</h4>
                    <ul style="list-style-type: none; padding-left: 0;">
                        <li>🕒 Prediction horizon: {days_ahead} days</li>
                        <li>📊 Confidence level: {confidence_level}</li>
                        <li>📅 Seasonal factors considered</li>
                        <li>📈 Historical patterns analyzed</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

# Add information about the system
st.sidebar.markdown('---')
st.sidebar.subheader('System Information')
st.sidebar.info('''
This energy demand prediction system:
- Uses advanced ML models (Random Forest, XGBoost, LightGBM)
- Provides predictions up to 2027
- Considers climate parameters
- Analyzes energy source distribution
- Includes long-term prediction capabilities
- Supports multiple cities in AP region
''')