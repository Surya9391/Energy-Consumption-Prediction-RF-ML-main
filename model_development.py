import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go

def model_development(final_data):
    # Separate features and target
    X = final_data.drop('Electricity Demand (MW)', axis=1)
    y = final_data['Electricity Demand (MW)']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42)
    }
    
    results = {}
    trained_models = {}
    feature_importance_dict = {}
    
    try:
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            trained_models[name] = model
            
            print(f"{name} Performance:")
            print(f"Root Mean Squared Error: {rmse:.2f}")
            print(f"Mean Absolute Error: {mae:.2f}")
            print(f"R2 Score: {r2:.4f}")
            
            # Store feature importance for tree-based models
            if name in ['Random Forest', 'XGBoost', 'LightGBM']:
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print("\nTop 10 Most Important Features:")
                print(feature_importance.head(10))
                
                # Store feature importance for later use
                feature_importance_dict[name] = feature_importance
        
        return trained_models, scaler, results, feature_importance_dict
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None, None

def plot_model_comparison(results):
    """
    Create a comparison plot of model performances
    """
    metrics = ['RMSE', 'MAE', 'R2']
    fig = go.Figure()
    
    for metric in metrics:
        values = [results[model][metric] for model in results.keys()]
        fig.add_trace(go.Bar(
            name=metric,
            x=list(results.keys()),
            y=values,
            text=[f'{v:.2f}' for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        xaxis_title='Models',
        yaxis_title='Metric Value'
    )
    
    return fig

def plot_feature_importance(feature_importance_df, model_name):
    """
    Create a feature importance plot for a specific model
    """
    fig = px.bar(feature_importance_df.head(10),
                 x='importance',
                 y='feature',
                 orientation='h',
                 title=f'Top 10 Feature Importance - {model_name}')
    
    return fig