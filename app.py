import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from get_data import get_data
from data_preprocessing import data_cleaning, calculate_heat_index, calculate_regional_heat_index
from model_development import train_models, plot_model_comparison, plot_feature_importance
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Energy Demand Prediction", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #2c3e50;
        padding: 10px 0;
        margin: 20px 0;
        border-bottom: 2px solid #eee;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: white !important;
        padding: 15px !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .sidebar-content {
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin: 10px 0;
    }
    .parameter-section {
        margin: 15px 0;
        padding: 15px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown(
    '<h1 class="main-header">Energy Demand Prediction System</h1>',
    unsafe_allow_html=True)

# Load and process data


@st.cache_data
def load_data():
    climate_data_path = "new datasets/Final_Energy_Prediction_Climate_Data_Corrected.csv"
    monthly_data_dir = "new datasets"
    climate_data, monthly_data = get_data(climate_data_path, monthly_data_dir)
    final_data = data_cleaning(climate_data, monthly_data)
    
    # Get the actual city columns from the processed data
    city_columns = [
        col for col in final_data.columns if col.startswith('City_')]
    cities = [col.replace('City_', '') for col in city_columns]
    
    return final_data, cities


# Load data and train models
if 'models' not in st.session_state:
    with st.spinner('Loading data and training models...'):
        final_data, cities = load_data()
        models, scaler, model_scores, feature_importance = train_models(
            final_data)
        
        st.session_state.models = models
        st.session_state.scaler = scaler
        st.session_state.scores = model_scores
        st.session_state.feature_importance = feature_importance
        st.session_state.cities = cities
        st.session_state.training_data = final_data
        st.session_state.feature_names = final_data.drop(
            'Electricity Demand (MW)', axis=1).columns.tolist()
        # Store the city columns separately
        st.session_state.city_columns = [
            col for col in st.session_state.feature_names if col.startswith('City_')]

# Model Performance Section
st.header('Model Performance Comparison')
col1, col2 = st.columns([2, 1])

with col2:
    # Model selection
    selected_model = st.selectbox(
        'Select Model for Prediction',
        ['Random Forest', 'XGBoost', 'LightGBM', 'ARIMA', 'LSTM'],
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
        """,
        'ARIMA': """
        - AutoRegressive Integrated Moving Average
        - Used for time series forecasting
        - Effective for capturing trends and seasonality
        """,
        'LSTM': """
        - Long Short-Term Memory
        - Effective for capturing long-term dependencies
        - Used for time series forecasting
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

# Sidebar inputs
st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
st.sidebar.markdown('### 📍 Location & Date Settings')
selected_city = st.sidebar.selectbox(
    'Select City', [
        'Kakinada', 'Surampalem', 'Rajahmundry', 'Samarlakota'])

# Date selection with better formatting
max_prediction_date = datetime(2027, 12, 31).date()
selected_date = st.sidebar.date_input(
    "Select Prediction Date",
    value=datetime.now().date(),
    min_value=datetime.now().date(),
    max_value=max_prediction_date
)

# Confidence indicator with improved styling
prediction_year = selected_date.year
confidence_level = (
    "High" if prediction_year == datetime.now().year else
    "Medium" if prediction_year == 2026 else
    "Low"
)
confidence_color = (
    "#28a745" if confidence_level == "High" else
    "#ffc107" if confidence_level == "Medium" else
    "#dc3545"
)
st.sidebar.markdown(f"""
<div style='padding: 15px; border-radius: 8px; background-color: {confidence_color}20;
    border-left: 4px solid {confidence_color}; margin: 10px 0;'>
    <h4 style='color: {confidence_color}; margin: 0;'>Prediction Confidence</h4>
    <p style='font-size: 1.2em; margin: 5px 0;'>{confidence_level}</p>
</div>
""", unsafe_allow_html=True)

# Climate parameters with improved organization
st.sidebar.markdown('### 🌡️ Climate Parameters')
with st.sidebar.container():
    temperature = st.slider('Temperature (°C)', 0.0, 50.0, 25.0)
    humidity = st.slider('Humidity (%)', 0.0, 100.0, 60.0)
    wind_speed = st.slider('Wind Speed (km/h)', 0.0, 50.0, 15.0)
    rainfall = st.slider('Rainfall (mm)', 0.0, 50.0, 5.0)

# Energy source parameters with better organization
st.sidebar.markdown('### ⚡ Energy Sources')
col1, col2 = st.sidebar.columns(2)

with col1:
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.markdown("#### 🌞 Renewable")
    solar_radiation = st.number_input('Solar (kWh/m²)', 0.0, 10.0, 1.0)
    wind_velocity = st.number_input('Wind (m/s)', 0.0, 30.0, 5.0)
    flow_rate = st.number_input('Hydro (m³/s)', 0.0, 1000.0, 100.0)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.markdown("#### 🏭 Conventional")
    thermal_energy = st.number_input(
        'Thermal (kJ)', 0.0, 10000000.0, 1000000.0)
    grid_capacity = st.number_input('Grid (kVA)', 0.0, 5000.0, 1000.0)
    st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content area
if st.sidebar.button('Generate Prediction', use_container_width=True):
    # Prepare input data for all 24 hours
    month = selected_date.month
    
    # Create predictions for all 24 hours
    hourly_predictions = []
    hourly_inputs = []

    # Define daily patterns for various parameters
    def get_hourly_temperature(base_temp, hour):
        # Temperature typically lowest at 4-5 AM, highest at 2-3 PM
        hourly_variation = {
            0: -2, 1: -2.5, 2: -3, 3: -3.5, 4: -4, 5: -3.5,
            6: -2, 7: -1, 8: 0, 9: 1, 10: 2, 11: 3,
            12: 3.5, 13: 4, 14: 4, 15: 3.5, 16: 3, 17: 2,
            18: 1, 19: 0, 20: -0.5, 21: -1, 22: -1.5, 23: -2
        }
        return base_temp + hourly_variation[hour]

    def get_hourly_solar_radiation(base_radiation, hour):
        # Solar radiation follows daylight pattern
        if 6 <= hour <= 18:  # Daylight hours
            # Peak at noon (hour 12)
            return base_radiation * (1 - abs(hour - 12) / 12)
        return 0  # No solar radiation at night

    def get_hourly_humidity(base_humidity, hour):
        # Humidity typically highest early morning, lowest afternoon
        hourly_variation = {
            0: 10, 1: 12, 2: 14, 3: 15, 4: 15, 5: 15,
            6: 14, 7: 12, 8: 8, 9: 4, 10: 0, 11: -2,
            12: -4, 13: -5, 14: -5, 15: -4, 16: -2, 17: 0,
            18: 2, 19: 4, 20: 6, 21: 7, 22: 8, 23: 9
        }
        new_humidity = base_humidity + hourly_variation[hour]
        return min(max(new_humidity, 0), 100)  # Keep between 0-100%

    def get_hourly_wind_speed(base_wind, hour):
        # Wind speed typically increases during day
        hourly_variation = {
            0: -2, 1: -2, 2: -2, 3: -2, 4: -1, 5: -1,
            6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 4,
            12: 5, 13: 5, 14: 5, 15: 4, 16: 3, 17: 2,
            18: 1, 19: 0, 20: -1, 21: -1, 22: -2, 23: -2
        }
        return max(base_wind + hourly_variation[hour], 0)

    for hour in range(24):
        # Initialize input data with all features including all city columns as
        # 0
        input_data = pd.DataFrame(
            0, index=[0], columns=st.session_state.feature_names)

        # Get hourly variations of parameters
        hourly_temp = get_hourly_temperature(temperature, hour)
        hourly_humidity = get_hourly_humidity(humidity, hour)
        hourly_wind = get_hourly_wind_speed(wind_speed, hour)
        hourly_solar = get_hourly_solar_radiation(solar_radiation, hour)

        # Set features with hourly variations
        regional_heat_index = calculate_regional_heat_index(
            hourly_temp, hourly_humidity, selected_city)
        temperature_humidity = hourly_temp * hourly_humidity / 100
        solar_efficiency = 100 * \
            (1 - hourly_humidity / 100) * (1 - rainfall / 25 if rainfall <= 25 else 0)
        wind_power_potential = hourly_wind**3
        is_summer = 1 if month in [4, 5, 6, 7, 8, 9] else 0
        is_peak_hour = 1 if hour in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] else 0

        # Calculate hourly power variations
        solar_power = hourly_solar * 100  # More power during daylight hours
        wind_power = (hourly_wind ** 3) * 3.33
        # Hydro and thermal remain relatively stable

        # Set all non-city features with hourly variations
        input_data['Temperature (°C)'] = hourly_temp
        input_data['Humidity (%)'] = hourly_humidity
        input_data['Wind Speed (km/h)'] = hourly_wind
        # Rainfall kept constant for the day
        input_data['Rainfall (mm)'] = rainfall
        input_data['Renewable_Share_%'] = ((solar_power + wind_power + flow_rate * 2) / (solar_power + wind_power + flow_rate *
                                               2 + thermal_energy * 0.0005 + grid_capacity * 0.4) * 100) if solar_power + wind_power + flow_rate * 2 > 0 else 0
        input_data['Thermal_Share_%'] = (thermal_energy * 0.0005 / (solar_power + wind_power + flow_rate * 2 +
                                             thermal_energy * 0.0005 + grid_capacity * 0.4) * 100) if solar_power + wind_power + flow_rate * 2 > 0 else 0
        input_data['Temperature_Humidity'] = temperature_humidity
        input_data['Regional_Heat_Index'] = regional_heat_index
        input_data['Solar_Efficiency'] = solar_efficiency
        input_data['Wind_Power_Potential'] = wind_power_potential
        input_data['Hour'] = hour
        input_data['Month'] = month
        input_data['Is_Summer'] = is_summer
        input_data['Is_Peak_Hour'] = is_peak_hour

        # Set the selected city to 1
        city_column = f'City_{selected_city}'
        if city_column in st.session_state.city_columns:
            input_data[city_column] = 1
        
        # Scale input data
        input_scaled = st.session_state.scaler.transform(input_data)
        
        # Make prediction for this hour
        prediction_result = st.session_state.models[selected_model].predict(
                input_scaled)
        # Handle different return types (array, scalar, Series)
        if hasattr(prediction_result, 'shape') and len(
                    prediction_result.shape) > 0 and prediction_result.shape[0] > 0:
                prediction = prediction_result[0]
        else:
                prediction = prediction_result  # If it's a scalar
        hourly_predictions.append(prediction)
        hourly_inputs.append(input_data.copy())

    # Create hourly prediction DataFrame
    prediction_df = pd.DataFrame({
        'Hour': range(24),
        'Predicted Demand (MW)': hourly_predictions
    })

    # Calculate average power generation for each source over 24 hours
    avg_solar_power = np.mean([get_hourly_solar_radiation(
        solar_radiation, h) * 100 for h in range(24)])

    # Calculate average wind power over 24 hours
    avg_wind_speeds = [get_hourly_wind_speed(wind_speed, h) for h in range(24)]
    avg_wind_power = np.mean([
        (0 if v < 3 else  # Cut-in speed
         0 if v > 25 else  # Cut-out speed
         150 if v > 15 else  # Rated power
         min(150, (v ** 3) * 0.44))  # Normal operation
        for v in avg_wind_speeds
    ])

    # Create energy source distribution with actual average values
    energy_sources = {
        'Solar Power': avg_solar_power,
        'Wind Power': avg_wind_power,
        'Hydro Power': flow_rate * 2,
        'Thermal Power': thermal_energy * 0.0005,
        'Grid Power': grid_capacity * 0.4
    }

    # Calculate demand statistics
    avg_demand = np.mean(hourly_predictions)
    max_demand = np.max(hourly_predictions)
    min_demand = np.min(hourly_predictions)

    # Define demand levels
    high_demand = np.percentile(hourly_predictions, 75)
    medium_demand = np.percentile(hourly_predictions, 50)
    low_demand = np.percentile(hourly_predictions, 25)

    # Results display with improved styling
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('### 📊 Demand Statistics')

    # Statistics in a grid with hover effects
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Demand", f"{avg_demand:.2f} MW")
        st.metric("High Demand", f"{high_demand:.2f} MW")
    with col2:
        st.metric("Peak Demand", f"{max_demand:.2f} MW")
        st.metric("Medium Demand", f"{medium_demand:.2f} MW")
    with col3:
        st.metric("Minimum Demand", f"{min_demand:.2f} MW")
        st.metric("Low Demand", f"{low_demand:.2f} MW")
    st.markdown('</div>', unsafe_allow_html=True)

    # Charts with improved containers
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('### 📈 24-Hour Forecast')

    # Create interactive line plot for 24-hour prediction
    fig = go.Figure()

    # Add prediction line
    fig.add_trace(
        go.Scatter(
            x=prediction_df['Hour'],
            y=prediction_df['Predicted Demand (MW)'],
            mode='lines+markers',
            name='Predicted Demand',
            hovertemplate='Hour: %{x}<br>Predicted Demand: %{y:.2f} MW<extra></extra>',
            line=dict(
                color='#1f77b4',
                width=2),
            marker=dict(
                size=8)))

    # Add reference lines for demand levels
    fig.add_hline(
        y=high_demand,
        line_dash="dash",
        line_color="red",
        annotation_text="High Demand Level",
        annotation_position="right")
    fig.add_hline(
        y=medium_demand,
        line_dash="dash",
        line_color="orange",
        annotation_text="Medium Demand Level",
        annotation_position="right")
    fig.add_hline(
        y=low_demand,
        line_dash="dash",
        line_color="green",
        annotation_text="Low Demand Level",
        annotation_position="right")

    # Add peak hours region
    peak_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    fig.add_vrect(
        x0=min(peak_hours) - 0.5, x1=max(peak_hours) + 0.5,
        fillcolor="rgba(255, 165, 0, 0.1)", layer="below", line_width=0,
        annotation_text="Peak Hours", annotation_position="top left"
    )

    # Update chart layout for better appearance
    fig.update_layout(
        title=None,  # Remove title as we have it in the container
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, l=40, r=40, b=40),
        xaxis=dict(
            tickmode='array',
            ticktext=[f'{i:02d}:00' for i in range(24)],
            tickvals=list(range(24)),
            gridcolor='rgba(128, 128, 128, 0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128, 128, 128, 0.2)',
        ),
        yaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128, 128, 128, 0.2)',
        ),
        height=500,
            )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Calculate historical demand
    def get_historical_demand(hour, month, is_weekend=False):
        # Base load pattern (MW)
        base_pattern = {
            0: 700, 1: 650, 2: 600, 3: 580, 4: 570, 5: 590,
            6: 650, 7: 750, 8: 850, 9: 950, 10: 1000, 11: 1050,
            12: 1100, 13: 1150, 14: 1100, 15: 1050, 16: 1000, 17: 950,
            18: 1000, 19: 1100, 20: 1000, 21: 900, 22: 800, 23: 750
        }

        # Season factors
        season_factor = 1.0
        if month in [12, 1, 2]:  # Winter
            season_factor = 1.2
        elif month in [3, 4, 5]:  # Spring
            season_factor = 0.9
        elif month in [6, 7, 8]:  # Summer
            season_factor = 1.3
        elif month in [9, 10, 11]:  # Fall
            season_factor = 0.95

        # Weekend reduction factor
        weekend_factor = 0.85 if is_weekend else 1.0

        return base_pattern[hour] * season_factor * weekend_factor

    # Calculate historical average
    is_weekend = selected_date.weekday() >= 5
    historical_avg = np.mean([get_historical_demand(
        h, selected_date.month, is_weekend) for h in range(24)])

    # Historical average with improved styling
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('### 📅 Historical Comparison')
    st.metric(
        "Historical Daily Average",
        f"{historical_avg:.2f} MW",
        delta=f"{((avg_demand - historical_avg)/historical_avg)*100:.1f}% vs Prediction"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Calculate total daily energy demand and supply
    total_daily_demand = sum(hourly_predictions)  # MWh (24 hours of MW = MWh)

    # Calculate total daily energy production from all sources
    total_daily_production = (
        avg_solar_power * 24 +  # Solar energy over 24 hours
        avg_wind_power * 24 +   # Wind energy over 24 hours
        flow_rate * 2 * 24 +    # Hydro energy over 24 hours
        thermal_energy * 0.0005 * 24 +  # Thermal energy over 24 hours
        grid_capacity * 0.4 * 24  # Grid energy over 24 hours
    )

    # Calculate energy balance
    energy_balance = total_daily_production - total_daily_demand
    energy_status = (
        "🟢 Energy Supply Exceeds Demand" if energy_balance > 100 else
        "🟡 Energy Supply Meets Demand" if abs(energy_balance) <= 100 else
        "🔴 Energy Supply Shortage"
    )

    # Display energy balance metrics with improved styling
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown('### ⚡ Daily Energy Balance')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Daily Demand",
            f"{total_daily_demand:.2f} MWh"
        )
    with col2:
        st.metric(
            "Total Daily Production",
            f"{total_daily_production:.2f} MWh"
        )
    with col3:
        st.metric(
            "Energy Balance",
            f"{abs(energy_balance):.2f} MWh",
            delta=f"{'Surplus' if energy_balance > 0 else 'Deficit'}"
        )

    # Display energy status with appropriate styling
    status_color = (
        "#28a745" if energy_balance > 100 else
        "#ffc107" if abs(energy_balance) <= 100 else
        "#dc3545"
    )
    st.markdown(f"""
    <div style='padding: 15px; border-radius: 8px; background-color: {status_color}20;
        border-left: 4px solid {status_color}; margin: 10px 0;'>
        <h4 style='color: {status_color}; margin: 0;'>Energy Status</h4>
        <p style='font-size: 1.2em; margin: 5px 0;'>{energy_status}</p>
        <p style='margin: 5px 0;'>
            {'Excess energy can be stored or exported' if energy_balance > 100 else
             'Energy supply is adequately meeting demand' if abs(energy_balance) <= 100 else
             'Additional energy sources or demand management required'}
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Energy distribution chart with improved container
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('### 🔋 Energy Source Distribution')

    # Create pie chart for energy source distribution with fixed colors
    colors = {
        'Solar Power': '#FFD700',  # Gold
        'Wind Power': '#87CEEB',   # Sky Blue
        'Hydro Power': '#4169E1',  # Royal Blue
        'Thermal Power': '#CD5C5C',  # Indian Red
        'Grid Power': '#808080'    # Gray
    }

    # Calculate total power and percentages
    total = sum(energy_sources.values())
    percentages = {k: (v / total) * 100 if total >
                   0 else 0 for k, v in energy_sources.items()}

    # Create pie chart
    pie_fig = go.Figure(data=[go.Pie(
        labels=list(energy_sources.keys()),
        values=list(energy_sources.values()),
        hole=.3,
        marker_colors=[colors[label] for label in energy_sources.keys()],
        hovertemplate="<b>%{label}</b><br>" +
        "Power: %{value:.1f} MW<br>" +
        "Share: %{percent:.1f}%<extra></extra>",
        textinfo='percent+label',
        textposition='outside'
    )])

    # Update pie chart layout
    pie_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, l=20, r=20, b=20),
        height=400,
    )
    st.plotly_chart(
        pie_fig,
        use_container_width=True,
        config={
            'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with system information
st.markdown("""
<div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 30px;'>
    <h3>ℹ️ System Information</h3>
    <ul style='list-style-type: none; padding-left: 0;'>
        <li>✓ Advanced ML models (Random Forest, XGBoost, LightGBM, ARIMA, LSTM)</li>
        <li>✓ Long-term predictions up to 2027</li>
        <li>✓ Comprehensive climate parameter analysis</li>
        <li>✓ Detailed energy source distribution</li>
        <li>✓ Support for multiple cities in AP region</li>
                    </ul>
                </div>
""", unsafe_allow_html=True)
